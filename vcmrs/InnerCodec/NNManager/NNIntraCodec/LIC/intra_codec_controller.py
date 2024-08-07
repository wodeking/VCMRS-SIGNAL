# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
from datetime import datetime
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from e2evc.opts import read_yaml
from e2evc.Utils import utils as e2eutils
from e2evc.Tools.compress_lossy_em import build_intra_codec, compress_one_image,decompress_one_image
from vcmrs.InnerCodec.NNManager.Main.cache_manager import API_func, ModelLoadingManager, log_cuda_stats, reset_cuda_stats

import vcmrs
# directory to store pretrained models
import vcmrs.InnerCodec.NNManager.Pretrained as pretrained
pretrained_dir = Path(pretrained.__path__[0]).joinpath('intra_codec')

QP_TO_EPOCH = { 
   22: 18, 
   27: 30,
   32: 120, 
   37: 170, 
   42: 220, 
   47: 270,
}

MODEL_TO_QP = { idx:k for idx, k in enumerate(QP_TO_EPOCH.keys()) }
QP_TO_MODEL = {v:k for k,v in MODEL_TO_QP.items()}
MODEL_TO_EPOCH= { idx:v for idx, v in enumerate(QP_TO_EPOCH.values()) }
WEIGHT_PATHS = {model_id: pretrained_dir.joinpath(f"model_{epoch}.pth.tar") for model_id, epoch in MODEL_TO_EPOCH.items()}

class IntraCodecController:
   """
      Stores the internal states related to the intra codec
   """
   def __init__(self, args, device="cuda", max_preloaded=3) -> None:
      
      self.__codec_model = None
      self.__full_qp_map = None
      self.model_loader = ModelLoadingManager(args, max_preloaded, weight_paths=WEIGHT_PATHS, build_func=build_intra_codec)
      self.device = device
      self.using_cuda = torch.cuda.is_available() and self.device != "cpu"
      
      # self.args = args
      # global cache_manager, error
      # cache_manager = IntraCacheManager(cache_fp="intra_cache.json", log_func=log_func)
      # error = utils.get_system_cfg().errorlog_func
   
   def get_model_index(self, qp):
      """
         Intrapolates the model index of the given QP value, using a predefined dict as anchors
         NOTE: Not a reasonable interpolation to fit a curve on model indexes, just an example
      """
      qp_keys = list(QP_TO_MODEL.keys())
      min_qp = min(qp_keys)
      max_qp = max(qp_keys)

      if isinstance(qp, str):
         m = re.search('(?:QP|qp)*[_]*(\d+)', qp) # E.g., QP32, qp32, QP_32, "32"
         qp = int(m.group(1))

      if qp <= min_qp: return QP_TO_MODEL[min_qp]
      if qp >= max_qp: return QP_TO_MODEL[max_qp]

      # Interpolation. Could be simply replaced with something like model_id = argmin(qp - MODEL_INDEXES.keys()).
      if self.__full_qp_map is None:
         full_qp_list = range(min_qp, max_qp + 1)
         full_idx_list = [np.nan] * len(full_qp_list)
         full_qp_map = { k: np.nan for k in range(min_qp, max_qp + 1)}
         for k,v in QP_TO_MODEL.items():
            full_qp_map[k] = v
         full_qp_list, full_idx_list = full_qp_map.keys(), full_qp_map.values()
         interpolated_idx = pd.Series(list(full_idx_list), dtype=np.float).interpolate(method='polynomial', order=2)
         full_idx_list = [round(n) for n in interpolated_idx]

         self.__full_qp_map = {k:v for k,v in zip(full_qp_list, full_idx_list)}

      return self.__full_qp_map[qp]
   
   '''Cached initialization of the intra codec models'''
   def switch_codec(self, QP=None, model_id=None):
      assert QP is not None or model_id is not None, "Neither QP or model_id is specified."
      # Override QP-to-model_id interpolation if model_id is given
      if model_id is None: 
         model_id = self.get_model_index(QP)

      rets = self.model_loader.get_model(model_id)
      self.__codec_model = rets["codec"]
      self.__codec_model.to(self.device, non_blocking=True)
      self.__codec_model.loaded_epoch = MODEL_TO_EPOCH[model_id]
      self.__codec_model.model_id = model_id
      return model_id
   
   @property
   def codec_model(self):
      if self.__codec_model is None: self.switch_codec(QP=0)
      return self.__codec_model
   
   @API_func
   def code_image(self, input_fp, output_bitstream_fp, output_image_fp, intra_cfg):
      """
         Performs a full compress - decompress cycle. Saves the bitstream and the decoded image to the given paths.

         Returns the bitstream bpp and the original image size.
      """
      s_time_primary = time.time()
      reset_cuda_stats(self)
      frame_qp = intra_cfg.video_qp
      model_id = self.switch_codec(QP = frame_qp)
      s_time_secondary = time.time()
      self.codec_model.time_tag = intra_cfg.time_tag # Must be after switch_codec()
      record = compress_one_image(
         self.codec_model, 
         i_imgfp=input_fp, 
         o_bsfp=output_bitstream_fp, 
      )
      vcmrs.log(f"{intra_cfg.time_tag} Intra NN coding done. Time = {(time.time() - s_time_secondary):.6}(s)")
      original_size, bpp = record["original_size"], record["bitstream_bpp"]

      s_time_secondary = time.time()

      # free GPU memory
      torch.cuda.empty_cache()

      decompress_one_image(
         self.codec_model, 
         i_bsfp = output_bitstream_fp, 
         o_imgfp = output_image_fp, 
         original_size = original_size,
         # frame_qp = MODEL_TO_QP[model_id], # used by intra human adapter
      )

      vcmrs.log(f"{intra_cfg.time_tag} Intra NN decoding done. Time = {(time.time() - s_time_secondary):.6}(s)")
      record["original_size"] = list(record["original_size"])
      
      record.update({
         "output_bitstream_fp": output_bitstream_fp,
         "output_image_fp": output_image_fp,
         "procedure": "full_code",
         "model_id": model_id,
         "frame_qp": MODEL_TO_QP[model_id],
         "timestamp": str(datetime.now())
      })
      
      vcmrs.log(f"{intra_cfg.time_tag} LIC coding done. BPP = {record['bitstream_bpp']}. Time = {(time.time() - s_time_primary):.6}(s)")
      log_cuda_stats(self)
      reset_cuda_stats(self)
      return record
   
   @API_func
   def decode_bitstream(self, input_fp, output_image_fp, intra_cfg):
      """
         Decompresses the given bitstream, saves the decoded image to the given path.
      """
      s_time = time.time()
      reset_cuda_stats(self)
      frame_qp = getattr(intra_cfg, "video_qp", None)
      model_id = getattr(intra_cfg, "model_id", None) # If given, overrides the model id from frame_qp
      model_id = self.switch_codec(QP=frame_qp, model_id=model_id)
      self.codec_model.time_tag = intra_cfg.time_tag
      original_size = 3, intra_cfg.picture_height, intra_cfg.picture_width
      decompress_one_image(
         self.codec_model, 
         i_bsfp = input_fp, 
         o_imgfp = output_image_fp, 
         original_size = original_size,
         # frame_qp = MODEL_TO_QP[model_id], # used by intra human adapter
      )
      # vcmrs.debug(f"Intra decompression done. Time = {(time.time() - s_time):.3}(s)")
      record = {
         "output_image_fp": output_image_fp,
         "procedure": "decode",
         "model_id": model_id,
         "frame_qp": MODEL_TO_QP[model_id],
         "timestamp": str(datetime.now())
      }
      vcmrs.log(f"{intra_cfg.time_tag} Intra frame decoding done. Time = {(time.time() - s_time):.6}(s)")
      log_cuda_stats(self)
      reset_cuda_stats(self)
      return record
   
   # @API_func
   # def encode_bitstream(self, input_fp, output_bitstream_fp):
   #    """
   #       Compresses the given image, saves the bitstream to the given path.
         
   #       Returns the bitstream bpp and the original image size.
   #    """
   #    s_time = time.time()
   #    rets = compress_one_image(self.codec_model, i_imgfp=input_fp, o_bsfp=output_bitstream_fp)
   #    _, bpp = rets["original_size"], rets["bitstream_bpp"]
   #    rets["original_size"] = list(rets["original_size"])
   #    rets.update({
   #       "output_bitstream_fp": output_bitstream_fp,
   #       "procedure": "encode",
   #       "timestamp": str(datetime.now())
   #    })
   #    vcmrs.log(f"Intra frame encoding done. BPP={bpp}. Time = {(time.time() - s_time):.6}(s)")
   #    return rets
