# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

from .nn_adapter import NNAdapter
import torch
import os
import vcmrs
def build_intra_human_adapter(args):
  rets = {}
  if args.model_fname is not None:
    model_fname=args.model_fname
    if os.path.exists(model_fname):
        if getattr(args.model_configs, "intra_human_adapter", None):
          device = 'cuda' if torch.cuda.is_available() else "cpu"
          intra_human_adapter = NNAdapter(config=args.model_configs.intra_human_adapter, device=device)
          intra_human_adapter.load_checkpoint(model_fname)
          intra_human_adapter.model.eval()
          rets["intra_human_adapter_model"] = intra_human_adapter
          vcmrs.debug(f'Intra human adapter constructed and weights loaded from {model_fname}')
    else:
      vcmrs.error(f"{model_fname} is not found. Terminating.")
      raise FileNotFoundError(model_fname)
  return rets

def build_inter_machine_adapter(args):
  rets = {}
  if args.model_fname is not None:
    model_fname=args.model_fname
    if os.path.exists(model_fname):
        if getattr(args.model_configs, "inter_machine_adapter", None):
          device = 'cuda' if torch.cuda.is_available() else "cpu"
          inter_machine_adapter = NNAdapter(config=args.model_configs.inter_machine_adapter, device=device)
          inter_machine_adapter.load_checkpoint(model_fname)
          inter_machine_adapter.model.eval()
          rets["inter_machine_adapter_model"] = inter_machine_adapter
          vcmrs.debug(f'Inter machie adapter constructed and weights loaded from {model_fname}')
    else:
      vcmrs.error(f"{model_fname} is not found. Terminating.")
      raise FileNotFoundError(model_fname)
  return rets


