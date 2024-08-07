# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
from argparse import ArgumentTypeError
from pathlib import Path
import os
import re
import torch
from e2evc.Utils import ctx
from torchvision.utils import save_image

from ..NNAdapter.adapter_utils import build_inter_machine_adapter
from .cache_manager import ModelLoadingManager, API_func, log_cuda_stats, reset_cuda_stats

from vcmrs.Utils import utils

pretrained_dir = Path(__file__).resolve().parent.parent.joinpath("Pretrained")
QP_TO_MODEL = {
    0:  0,
   -1: -1 # -1 indicates fallback mode id
}

WEIGHT_PATHS = {
   0: pretrained_dir.joinpath("inter_machine_adapter_A3_wo_inject_w01.pth.tar"),
   -1: pretrained_dir.joinpath("inter_machine_adapter_A0_w001.pth.tar"),
}
class InterMachineAdapterController:
   """
      Stores the internal states related to the inter machine adapter
   """
   def __init__(self, args, device="cuda", max_preloaded=3, log_func=lambda x:x) -> None:
      
      self.device = device
      self.args = args
      self.__inter_machine_adapter_model = None
      self.using_cuda = torch.cuda.is_available() and self.device != "cpu"
      self.model_loader = ModelLoadingManager(args, max_preloaded, weight_paths=WEIGHT_PATHS, build_func=build_inter_machine_adapter)
   
   def get_model_index(self, qp):
      """
         Intrapolates the model index of the given QP value, using a predefined dict as anchors
         currently using just two models, one for normal case and one for fallback mode
      """
      qp_keys = list(QP_TO_MODEL.keys())

      if isinstance(qp, str):
         m = re.search('(?:QP|qp)*[_]*(\d+)', qp) # E.g., QP32, qp32, QP_32, "32"
         qp = int(m.group(1))

      min_qp = min(qp_keys)
      max_qp = max(qp_keys)

      # Model selection rules
      if qp <= min_qp: return QP_TO_MODEL[min_qp]
      if qp >= max_qp: return QP_TO_MODEL[max_qp]

      return QP_TO_MODEL[0] # One size fit all for now

   
   '''Lazy initialization of the inter machine adapter model'''
   def switch_model(self, QP, intra_fallback):
      if intra_fallback:
         self.model_loader.args.model_configs.inter_machine_adapter.injections=[1, 1]
         model_id = self.get_model_index(-1)
      else:
         self.model_loader.args.model_configs.inter_machine_adapter.injections=[0, 0]
         model_id = self.get_model_index(QP)

      rets = self.model_loader.get_model(model_id)
      self.__inter_machine_adapter_model = rets["inter_machine_adapter_model"]
      # self.log(f'Activated codec {model_id}. Memory usage: '
      #          f'Allocated: {torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB, '
      #          f'Cached: {torch.cuda.memory_reserved(self.device)/1024**3:.2f} GB')
      return model_id
   
   @property
   def inter_machine_adapter_model(self):
      if self.__inter_machine_adapter_model is None: self.switch_model(0, False)
      return self.__inter_machine_adapter_model

   @API_func
   @torch.no_grad()
   def apply_inter_machine_adapter(self, input_fp, output_image_fp, video_info, param):
      """
         Apply the inter machine adapter to the given input image path, saves the output image to the given 
         output file path.
      """
      reset_cuda_stats(self)
      model_id = self.switch_model(param.qp, param.intra_fallback)

      conv_ctx  = ctx.int_conv("none") # Float convs
      if utils.getattr_by_path(self.args, "model_configs.inter_machine_adapter.int_conv", False):
         conv_ctx = ctx.int_conv()

      with conv_ctx:
         if os.path.splitext(input_fp)[1].lower()=='.png':
           rets = self.inter_machine_adapter_model.apply_adapter_rgb(input_fp, param.qp, original_size=None)
         else:
           rets = self.inter_machine_adapter_model.apply_adapter_yuv(input_fp, video_info, param)

      save_image(rets, output_image_fp)

      log_cuda_stats(self)
      reset_cuda_stats(self)
      return {
         "input_fp": input_fp,
         "output_image_fp": output_image_fp,
         "model_id": model_id
      }
