# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import torch
import cv2

from ...NNManager.Main.cache_manager import API_func
import vcmrs

# directory to store pretrained models
import vcmrs.InnerCodec.NNManager.Pretrained as pretrained
pretrained_dir = pretrained.__path__[0]

class IntraCodecController:
   """Stores the internal states related to the intra codec

   Args:
     args: input arguments
     device: computation device, for example 'cuda' or 'cpu'
   """
   def __init__(self, args, device="cuda") -> None:
      self.__codec_model = None
      self.device = device
      self.using_cuda = torch.cuda.is_available() and self.device != "cpu"

   def get_model_id(self, qp):
     '''Get model id from QP value
 
     Args:
       qp: intra qp value
     '''
     if qp > 32:
       return 1
     return 0

   def get_frame_qp(self, model_id):
     '''Get frame QP from model id
     This value is used by intra human adapter

     Args: 
       model_id: model id
     '''
     if model_id == 1: return 42
     return 27


   @API_func
   def code_image(self, input_fp, output_bitstream_fp, output_image_fp, intra_cfg):
      """Performs a full compress - decompress cycle. Saves the bitstream and the decoded image to the given paths.

      Args: 
        input_fp: input image file name
        output_bitstream_fp: output bitstream file name
        output_image_fp: reconstructed image file name
        intra_cfg: configuration for the intra codec
          intra_cfg.video_qp: QP setting for the video

        Returns the bitstream bpp and the original image size.
      """
      frame_qp = intra_cfg.video_qp
      model_id = self.get_model_id(frame_qp)

      ret = {}
      # a dummy implmentation, save a downsampled png file as the bitstream
      img = cv2.imread(str(input_fp))
      H,W,C = img.shape
      img2 = cv2.resize(img, (W//(2**(model_id+2)), H//(2**model_id+2)))
      bs = bytearray(cv2.imencode('.png', img2)[1].tobytes())

      with open(output_bitstream_fp, 'wb') as f:
        f.write(bs)
      recon_img = cv2.resize(img2, (W, H))
      cv2.imwrite(output_image_fp, recon_img)

      # prepare output
      ret['original_size'] = (W, H)
      ret['bitstream_bpp'] = len(bs) / W / H
      ret["output_bitstream_fp"] = output_bitstream_fp
      ret["output_image_fp"] = output_image_fp
      ret['precedure'] = 'full_code'
      ret['model_id'] = model_id
      ret['frame_qp'] = self.get_frame_qp(model_id)
      ret['timestamp'] = str(datetime.now())

      vcmrs.debug('Intra codec encoding completed')
      vcmrs.debug(ret)
      return ret
   
   @API_func
   def decode_bitstream(self, input_fp, output_image_fp, intra_cfg):
      """
         Decompresses the given bitstream, saves the decoded image to the given path.

      Args: 
         input_fp: input bitstream file name
         output_image_fp: reconstructed image file name
         intra_cfg: configurations for the intra codec
           intra_cfg.picture_width, intra_cfg.picture_height: wight and height of the reconstructed picture
           intra_cfg.model_id: the model id used in encoding
      """
      frame_qp = getattr(intra_cfg, "video_qp", None)
      model_id = getattr(intra_cfg, "model_id", None) 
      original_size = 3, intra_cfg.picture_height, intra_cfg.picture_width

      with open(input_fp, 'rb') as f:
        bs = f.read()

      img2 = cv2.imdecode(np.frombuffer(bs, dtype=np.int8), -1)
      recon_img = cv2.resize(img2, (intra_cfg.picture_width, intra_cfg.picture_height))
      cv2.imwrite(output_image_fp, recon_img)

      ret = {
         "output_image_fp": output_image_fp,
         "procedure": "decode",
         "model_id": model_id,
         "frame_qp": self.get_frame_qp(model_id),
         "timestamp": str(datetime.now())
      }

      vcmrs.debug('Intra codec decoding completed')
      vcmrs.debug(ret)
      return ret

