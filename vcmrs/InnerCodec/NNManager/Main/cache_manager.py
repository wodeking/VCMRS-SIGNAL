# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import gc
from pathlib import Path
import torch
import functools
import traceback
from vcmrs.Utils import utils
import vcmrs

class ModelLoadingManager:
   def __init__(self, args, max_preloaded, weight_paths, build_func) -> None:
      assert max_preloaded >= 0, "Non negative number of max preloaded models expected"
      self.max_preloaded = max_preloaded
      self.__loaded_model = {} # First in first out queue
      self.args = args
      self.weight_paths = weight_paths
      self.build_func = build_func

   def get_model(self, model_index):
      """
         Load or return the preloaded model of the given index. Unload the oldest preloaded model if necessary.
      """
      if model_index in self.__loaded_model:
         # Move to the end of queue as if just loaded (will live longer)
         rets = self.__loaded_model.pop(model_index) 
         self.__loaded_model[model_index] = rets
      else:
         # Load if not preloaded
         if len(self.__loaded_model) == self.max_preloaded:
            # Unload if needed
            oldest_key = list(self.__loaded_model.keys())[0]
            del self.__loaded_model.pop(oldest_key)['codec']

         self.args.model_fname = self.weight_paths[model_index]
         rets = self.build_func(self.args)
         self.__loaded_model[model_index] = rets

      return rets

def API_func(f, success_code=0):
   """
      Wrapper decoration to return a dict of `error_code` of the request along with the returned `values`. 
      If exception occurs, `error_code` is the title of the error and `values` contains the call stack.
   """
   @functools.wraps(f)
   def wrapper(*args, **kwargs):
      error_code, values = None, None
      try:
         values = f(*args, **kwargs)

         error_code = success_code

      except Exception as e:
         callstack = f"\n-------Exception info-----------\n{traceback.format_exc()}"
         callstack += "\n--------Call stack (most recent last)----------\n'" + '\n'.join([line.strip() for line in traceback.format_stack()])
         #cfg.errorlog_func(callstack)
         vcmrs.error(callstack)
         error_code = str(e),
         values = callstack
         raise e
      finally:
         return {
                  "error": error_code,
                  "values": values
               }
   return wrapper

def empty_cache_memory():
   gc.collect()
   torch.cuda.empty_cache()

def reset_cuda_stats(obj):
   if obj.using_cuda:
      total = torch.cuda.get_device_properties(obj.device).total_memory
      reserved = torch.cuda.memory_reserved(obj.device)
      if (ratio:=reserved/total) > 0.5:
         vcmrs.debug(F"Memory usage of {reserved/1024**3:.2f} GB ({(ratio*100):.2f}%) exceeds the safe threshold. Releasing cached GPU memory.")
         empty_cache_memory()
      torch.cuda.reset_peak_memory_stats(obj.device)

def log_cuda_stats(obj):
  if obj.using_cuda:
   vcmrs.debug(f'Peak memory usage ({type(obj).__name__}): '
         f'Allocated: {torch.cuda.max_memory_allocated(obj.device)/1024**3:.2f} GB, '
         f'Cached: {torch.cuda.max_memory_reserved(obj.device)/1024**3:.2f} GB')
 
