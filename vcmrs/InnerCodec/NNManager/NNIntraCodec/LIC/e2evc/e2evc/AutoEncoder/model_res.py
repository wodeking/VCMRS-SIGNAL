# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
from torch import nn
from e2evc.Utils import utils_nn
from e2evc.Utils.utils import getattr_by_path
# from antialiased_cnns import BlurPool

class EncoderRes(nn.Module):
  ''' Encoder with ResBlock model
  '''

  def __init__(self, 
      input_channels=3, 
      output_channels=128,
      mid_channels=128):
    '''Initialize the encoder

    Parameters
    ----------------
    intput_channels : int
      number of input channels
    output_channels : int
    mid_channels : int
    '''
    super().__init__()

    self.network = nn.Sequential(    
      utils_nn.ResBlock(input_channels, mid_channels, stride=2),
      utils_nn.ResBlock(mid_channels, mid_channels),
      
      utils_nn.ResBlock(mid_channels, mid_channels, stride=2),
      utils_nn.ResBlock(mid_channels, mid_channels),
      utils_nn.IntConv2d(mid_channels, output_channels, kernel_size=3, stride=2, padding=1))
      

  def forward(self, x):
    '''overide forward function from base class
    '''
    x = self.network(x)
    return x

class DecoderRes(nn.Module):
  '''Decoder
  '''

  def __init__(self, \
      input_channels=128, \
      output_channels=3, \
      mid_channels=128):
    '''Initialize the decoder

    Parameters
    ----------------
    input_channels : int
      number of input channels
    output_channels : int
      number of output channels
    mid_channels
    '''
    super().__init__()
    up_module = utils_nn.PSConv2d
    self.network = nn.Sequential( 
      utils_nn.ResBlock(input_channels, mid_channels),
      utils_nn.ResBlockUp(mid_channels, mid_channels,up_module),

      utils_nn.ResBlock(mid_channels, mid_channels),
      utils_nn.ResBlockUp(mid_channels, mid_channels,up_module),

      utils_nn.ResBlock(mid_channels, mid_channels),
      up_module(mid_channels, output_channels),
      )

  def forward(self, x):
    '''overide forward function from base class
    '''
    x = self.network(x)
    return x

