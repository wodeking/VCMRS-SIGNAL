# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
from torch import nn
import torch.nn.functional as F
from ...Utils.utils_nn import IntConv2d

# convolution
class BasicConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = IntConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        x = self.conv(x)
        return x

# conv block with activation
# this is not used by anyone
class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = IntConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.PReLU()
    def forward(self, x):
        x = self.conv(x)
        return self.act(x)

# 1x1 conv
class Conv1x1(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv = IntConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        return  self.conv(x)


# res block without the output activation
#class ResBlock(nn.Module):
#    def __init__(self, in_channels, nr_filters):
#        super().__init__()
#        self.conv1 = BasicConv(in_channels, nr_filters)
#        self.act = nn.PReLU(nr_filters)
#        self.conv2 = BasicConv(nr_filters, nr_filters)
#    
#    def forward(self, x):
#        orig_x = x
#        x = self.conv1(x)
#
#        x = self.act(x)
#        x = self.conv2(x)
#        return x + orig_x

# res block with the output activation
class ResBlockA(nn.Module):
  def __init__(self, in_channels, nr_filters):
    super().__init__()
    self.conv1 = BasicConv(in_channels, nr_filters)
    self.act1 = nn.PReLU(nr_filters)
    self.conv2 = BasicConv(nr_filters, nr_filters)
    self.act2 = nn.PReLU(nr_filters)
  
  def forward(self, x):
    orig_x = x
    x = self.conv1(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.act2(x + orig_x)
    return x

# This function handle odd size of the high resolution representations
def pad_if_needed(x, x0_size, mode='replicate'):
    pad=[0, x0_size[-1] - x.size(-1), 0, x0_size[-2] - x.size(-2)]
    x = F.pad(x, pad, mode=mode)
    return x


class Upsampler(nn.Module):
    def __init__(self,
        in_channels,
        scale=2):
        super().__init__()
        self.conv = BasicConv(in_channels, in_channels*scale*scale)
        self.pixelshuffler = nn.PixelShuffle(scale)
    
    def forward(self, x):
      x = self.conv(x)
      x = self.pixelshuffler(x)
      return x

class RoundQuantizer(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels
    
    def forward(self, x_):
        x = x_.round()
        round_bits = (x_*self.levels + self.levels/2 - x*self.levels).round() % self.levels
        return x, round_bits

class RoundDequantizer(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels

    def forward(self, x, round_bits):
        x_ = (round_bits - 2) / self.levels + x
        return x_

class FixedPooling(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale=2

    def forward(self, x):
        # x : NCHW
        x = x[:, :, self.scale-1::self.scale, self.scale-1::self.scale]
        # return interger_x, float_x, and round_bits, to be consistent with AvgPooling
        return x, x, 0

class AvgPooling(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
        # Quantizer
        self.round_quantizer = RoundQuantizer()
    
    def forward(self, x):
        x_ = self.pool(x)
        x, round_bits = self.round_quantizer(x)
        # return interger_x, float_x, and round_bits
        return x, x_, round_bits

