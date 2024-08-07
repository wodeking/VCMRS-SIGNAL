# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import torch
import torch.nn.functional as F

# pad an input tensor to be divisible by a number
# padding are to the bottom and right
# input: 
#   x: tensor of shape .....HW
def pad_tensor(x, divisor, mode='constant'):
  # calculate padding size
  h,w = map(int, x.shape[-2:])
  pad_h = -(-h // divisor) * divisor - h 
  pad_w = -(-w // divisor) * divisor - w 
  return F.pad(x, (0, pad_w, 0, pad_h), mode)

# remove padding from bottom and right
def remove_padding(x, orig_size):
  return x[..., :orig_size[-2], :orig_size[-1]]


