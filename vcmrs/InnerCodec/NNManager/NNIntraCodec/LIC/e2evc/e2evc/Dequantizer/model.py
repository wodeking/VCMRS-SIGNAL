# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
from torch import nn
from torch.autograd import Function
from e2evc.Utils import utils

# Uniform dequantization module
class UniDequantizer(nn.Module):
  '''Uniform dequantizer 
  '''

  def __init__(self, levels=256):
    '''
    Parameters
    -----------------
    levels : levels of the input int 
    '''
    super(UniDequantizer, self).__init__()

    self.levels = levels

  def forward(self, x, min_val=-1, max_val=1):
    '''uniform quantization
    '''
    utils.validate_input_int(x, self.levels)

    delta = (max_val - min_val)/self.levels
    x = x * delta + delta/2 + min_val
    return x


# Uniform dequantization module
class UniDequantizerDelta(nn.Module):
  '''Uniform dequantizer given delta
  '''

  def __init__(self, delta=1):
    '''
    Parameters
    -----------------
    levels : levels of the input int 
    '''
    super().__init__()

    self.delta = delta

  def forward(self, x):
    '''uniform quantization
    '''
    x = x * self.delta
    return x
