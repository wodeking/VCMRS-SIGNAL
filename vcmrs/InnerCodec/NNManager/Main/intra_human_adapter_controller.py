# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
from argparse import ArgumentTypeError
from pathlib import Path
import re
import torch
from torchvision.utils import save_image
from e2evc.Utils import ctx

from ..NNAdapter.adapter_utils import build_intra_human_adapter
from .cache_manager import API_func, ModelLoadingManager, log_cuda_stats, reset_cuda_stats

pretrained_dir = Path(__file__).resolve().parent.parent.joinpath("Pretrained")
QP_TO_MODEL = {
   0: 0
}

WEIGHT_PATHS = {
   0: pretrained_dir.joinpath("intra_human_adapter_A1.pth.tar"),
}

class IntraHumanAdapterController:
   """
      Stores the internal states related to the intra human adapter
   """
   def __init__(self, args, device="cuda", max_preloaded=3) -> None:
      
      self.device = device
      self.args = args
      self.__intra_human_adapter_model = None
      self.using_cuda = torch.cuda.is_available() and self.device != "cpu"
      self.model_loader = ModelLoadingManager(args, max_preloaded, weight_paths=WEIGHT_PATHS, build_func=build_intra_human_adapter)
   
   def get_model_index(self, qp):
      """
         Intrapolates the model index of the given QP value, using a predefined dict as anchors
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

   '''Cached initialization of the intra human adapter model'''
   def switch_model(self, QP):
      model_id = self.get_model_index(QP)
      rets = self.model_loader.get_model(model_id)
      self.__intra_human_adapter_model = rets["intra_human_adapter_model"]

      return model_id
   
   @property
   def intra_human_adapter_model(self):
      if self.__intra_human_adapter_model is None: self.switch_model(0)
      return self.__intra_human_adapter_model

   @API_func
   @torch.no_grad()
   @ctx.int_conv()
   def apply_intra_human_adapter(self, input_image, frame_qp):
      """
         Apply the intra human adapter to the given input image, returns the adapted image.
      """
      reset_cuda_stats(self)
      model_id = self.switch_model(frame_qp)
      adapted_image = self.intra_human_adapter_model.apply_adapter_rgb(
        input_fp=input_image, 
        frame_qp=frame_qp, 
        original_size=None)

      log_cuda_stats(self)
      reset_cuda_stats(self)
      return {
         "adapted_image": adapted_image,
         "model_id": model_id
      }
