# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
# Context

import torch
from contextlib import ContextDecorator
import numpy as np

# global variable for contexts
_mode = 'none'

# total precision
_precision = 1

# high precision for mixture mode
_high_precision = 1

def _set_mode(mode):
  global _precision, _high_precision, _mode

  if mode == 'none':
    _precision = 0
  elif mode == 'float32':
    _precision = 2**(23+1)
  elif mode == 'float64':
    _precision = 2**(52+1)
  elif mode == 'mixture':
    _precision = 2**(23+1)
    _high_precision = 2**(52+1) 
  else:
    assert False, 'unsupported int_conv mode'

  _mode = mode
  torch.backends.cudnn.enabled = mode=='none'


class int_conv(ContextDecorator):

  def __init__(self, mode='float32'): 
    '''
      mode: none / float32 / float64 / mixture
      when mixture mode is enabled, integer convolution may be performed in a mixture of high and low precisions. some operations are in float64 while other operations are in float32
    '''
    self.mode = mode

  def __enter__(self):
    self.previous_mode = _mode # Precaution for nested contexts
    _set_mode(self.mode)

  def __exit__(self, exc_type, exc, exc_tb):
    _set_mode(self.previous_mode)
    
