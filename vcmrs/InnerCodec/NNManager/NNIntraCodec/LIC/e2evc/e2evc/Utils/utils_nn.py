# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet
from . import ctx

# calculate ouptut padding
#https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/3
def cal_output_padding(i, s, p, k):
  output_padding = i*2 - ((i-1)*s - 2*p + k)
  return output_padding

# validate input data
def validate_input_int(x, levels=256, normalize=True):
    """ validate input range in x should be int in float32 type
    """
    assert x.dtype==torch.float32 and torch.norm(torch.round(x) - x,2)<1E-8, "Input should be int in float32 type"
    
    assert levels==256, "Only 256 is supported at the moment"
    assert torch.min(x)>=0 and torch.max(x)<levels, f"Input should be between 0 and {levels-1}"
    if normalize: 
      x = (x / (levels-1) - 0.5) * 2

    return x

  
class IntConv2d(nn.Conv2d):
  def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size, 
        stride = 1, 
        padding = 0, 
        dilation = 1, 
        groups: int = 1, 
        bias: bool = True, 
        padding_mode: str = 'zeros', 
        device=None, 
        dtype=None) -> None:
      super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
      self.is_quantized = False

  # prepare quantized weights
  def quantize(self):
    if not self.is_quantized:
        if self.bias is None:
            self.float_bias = torch.zeros(self.out_channels, device=self.weight.device)
        else:
            self.float_bias = self.bias.detach().clone()

        #sf const 
        sf_const = 48

        N = np.prod(self.weight.shape[1:])
        self.N = N
        self.factor = np.sqrt(ctx._precision)
        #self.sf = 1/6 #precision bits allocation factor
        self.sf = np.sqrt(sf_const/N)

        # perform the calculate ion CPU to stabalize the calculation
        self.w_sum = self.weight.cpu().abs().sum(axis=[1,2,3]).to(self.weight.device)
        self.w_sum[self.w_sum==0] = 1 # prevent divide by 0

        self.fw = (self.factor / self.sf -  np.sqrt(N/12)*5) / self.w_sum
        if ctx._mode=='mixture':
          self.sig_channels = 4
          self.sig_threshold = 500 # apply significant conv when the x.abs().max()*N/w_sum is greater than this threshold
          #self.sig_sf = np.sqrt(sf_const/(self.sig_channels*np.prod(self.weight.shape[2:])))
          self.sig_sf = self.sf

          # calculate sig_fw
          self.sig_factor = np.sqrt(ctx._high_precision)
          self.sig_fw = (self.sig_factor / self.sig_sf -  np.sqrt(N/12)*5)/ self.w_sum 

          #intify sig weights
          self.sig_weight = torch.round(self.weight.detach().clone().double() * \
            self.sig_fw.view(-1, 1, 1, 1)).double()

        # intify weights
        self.weight.copy_(torch.round(self.weight.detach().clone() * self.fw.view(-1, 1, 1, 1)))
 
        # set bias to 0
        if self.bias is not None:
            self.bias.zero_()

        self.is_quantized = True

  def forward(self, x):
    if ctx._mode == 'none':
      return super().forward(x)
    else:
      self.quantize()

    # convert to float 64
    if ctx._mode == 'float64':
      x = x.double()
      self = self.double()

    # Calculate factor
    fx = 1

    x_abs = x.abs()
    x_max = x_abs.max()
    if x_max > 0:  
      fx = (self.factor * self.sf-0.5) / x_max

    # perform conv2d
    if ctx._mode=='mixture' and (x_max*self.N/self.w_sum.max() > self.sig_threshold):
       # mixture precision mode
       x_abs_max = x_abs.max(axis=-1)[0].max(axis=-1)[0].max(axis=0)[0]
       #sorted_idx = reversed(torch.argsort(x_abs_max, stable=True))
       _, sorted_idx = torch.sort(x_abs_max, descending=True, stable=True)
       sig_idx = sorted_idx[:self.sig_channels]
       insig_idx = sorted_idx[self.sig_channels:]
       sig_fx = (self.sig_factor * self.sig_sf - 0.5) / x_max
       
       # computer sig
       # intify sig_conv
       x_sig = x[:, sig_idx, :, :].double()
       x_sig = torch.round(sig_fx * x_sig)
       x_sig_out = torch.nn.functional.conv2d(
         x_sig, 
         self.sig_weight[:, sig_idx, :, :],
         bias=None, 
         stride=self.stride, 
         padding=self.padding,
         dilation=self.dilation,
         groups=self.groups)

       # perform convolution
       x_sig_out /= sig_fx * self.sig_fw.view(-1, 1, 1)
       x_sig_out = x_sig_out.float()

       # handle insignificant channels
       insig_weight = self.weight[:, insig_idx, :, :]
       x_insig = torch.round(fx * x[:, insig_idx, :, :])
       x_insig_out = torch.nn.functional.conv2d(
         x_insig, 
         insig_weight,
         bias=None, 
         stride=self.stride, 
         padding=self.padding,
         dilation=self.dilation,
         groups=self.groups)

       x_insig_out /= fx * self.fw.view(-1, 1, 1)
       x = x_insig_out + x_sig_out
       
    else:   

      # intify x
      x = torch.round(fx * x)
      x = super().forward(x)

      # x should be all integers
      x /= fx * self.fw.view(-1, 1, 1)
      x = x.float()

    # apply bias in float format
    x = (x.permute(0,2,3,1)+self.float_bias).permute(0,3,1,2).contiguous()

    return x

class IntTransposedConv2d(nn.ConvTranspose2d):
  def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, output_padding = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)
    
    self.is_quantized = False

  # prepare quantized weights
  def quantize(self):
    if not self.is_quantized:
        if self.bias is None:
            self.float_bias = torch.zeros(self.out_channels, device=self.weight.device)
        else:
            self.float_bias = self.bias.detach().clone()

        #sf const 
        sf_const = 48

        N = np.prod(self.weight.shape) / self.weight.shape[1] # (in, out, kH, kW)
        self.N = N
        self.factor = np.sqrt(ctx._precision)
        #self.sf = 1/6 #precision bits allocation factor
        self.sf = np.sqrt(sf_const/N)

        # perform the calculate ion CPU to stabalize the calculation
        self.w_sum = self.weight.cpu().abs().sum(axis=[0,2,3]).to(self.weight.device)
        self.w_sum[self.w_sum==0] = 1 # prevent divide by 0

        self.fw = (self.factor / self.sf -  np.sqrt(N/12)*5) / self.w_sum

        if ctx._mode=='mixture':
          self.sig_channels = 4
          self.sig_threshold = 500 # apply significant conv when the x.abs().max()*N/w_sum is greater than this threshold
          #self.sig_sf = np.sqrt(sf_const/(self.sig_channels*np.prod(self.weight.shape[2:])))
          self.sig_sf = self.sf

          # calculate sig_fw
          self.sig_factor = np.sqrt(ctx._high_precision)
          self.sig_fw = (self.sig_factor / self.sig_sf -  np.sqrt(N/12)*5)/ self.w_sum 

          #intify sig weights
          self.sig_weight = torch.round(self.weight.detach().clone().double() * \
            self.sig_fw.view(1, -1, 1, 1)).double()

        # intify weights
        self.weight.copy_(torch.round(self.weight.detach().clone() * self.fw.view(1, -1, 1, 1)))
 
        # set bias to 0
        if self.bias is not None:
            self.bias.zero_()

        self.is_quantized = True

  def forward(self, x):
    if ctx._mode == 'none':
      return super().forward(x)
    else:
      self.quantize()

    # convert to float 64
    if ctx._mode == 'float64':
      x = x.double()
      self = self.double()

    # Calculate factor
    fx = 1

    x_abs = x.abs()
    x_max = x_abs.max()
    if x_max > 0:  
      fx = (self.factor * self.sf-0.5) / x_max

    # perform conv2d
    if ctx._mode=='mixture' and (x_max*self.N/self.w_sum.max() > self.sig_threshold):
       # mixture precision mode
       x_abs_max = x_abs.max(axis=-1)[0].max(axis=-1)[0].max(axis=0)[0]
       sorted_idx = reversed(torch.argsort(x_abs_max))
       sig_idx = sorted_idx[:self.sig_channels]
       insig_idx = sorted_idx[self.sig_channels:]
       sig_fx = (self.sig_factor * self.sig_sf - 0.5) / x_max
       
       # computer sig
       # intify sig_conv
       x_sig = x[:, sig_idx, :, :].double()
       x_sig = torch.round(sig_fx * x_sig)
       x_sig_out = torch.nn.functional.conv_transpose2d(
         x_sig, 
         self.sig_weight[sig_idx, :, :, :],
         bias=None, 
         stride=self.stride, 
         padding=self.padding,
         output_padding=self.output_padding,
         dilation=self.dilation,
         groups=self.groups)

       # perform convolution
       x_sig_out /= sig_fx * self.sig_fw.view(-1, 1, 1)
       x_sig_out = x_sig_out.float()

       # handle insignificant channels
       insig_weight = self.weight[insig_idx, :, :, :]
       x_insig = torch.round(fx * x[:, insig_idx, :, :])
       x_insig_out = torch.nn.functional.conv_transpose2d(
         x_insig, 
         insig_weight,
         bias=None, 
         stride=self.stride, 
         padding=self.padding,
         output_padding=self.output_padding,
         dilation=self.dilation,
         groups=self.groups)

       x_insig_out /= fx * self.fw.view(-1, 1, 1)
       x = x_insig_out + x_sig_out
       
    else:   

      # intify x
      x = torch.round(fx * x)
      x = super().forward(x)

      # x should be all integers
      x /= fx * self.fw.view(-1, 1, 1)
      x = x.float()

    # apply bias in float format
    x = (x.permute(0,2,3,1)+self.float_bias).permute(0,3,1,2).contiguous()

    return x

# generate one hot vectors from labels
def one_hot(labels, C, weighted=False):
  if weighted:
    y = torch.arange(C).repeat(labels.shape[0], 1).cuda()
    y2 = labels.reshape(labels.shape[0], 1).cuda().expand_as(y)
    x = 1 - torch.abs(y - y2)/(C-1)
  else:
    x = torch.zeros(labels.shape[0], C).cuda()
    x[torch.arange(labels.shape[0]), labels.type(torch.long)] = 1 
  return x

class IntBasicBlock(resnet.BasicBlock):
  def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer = None) -> None:
    super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
    self.conv1 = IntConv2d(inplanes, inplanes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    self.conv2 = IntConv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# BatchNormal to replace default BatchNormalization layer
class BN(nn.Module):
  def __init__(self, channels, type='None', n_groups=16):
    ''' batch normalization layer
    '''
    super().__init__()
    if type=='bn1':
      self.bn = nn.BatchNorm2d(channels, momentum=None, track_running_stats=True)
    elif type=='bn2':
      self.bn = nn.BatchNorm2d(channels, momentum=0, track_running_stats=False)
    elif type=='instance':
      self.bn = nn.InstanceNorm2d(channels, momentum=0, track_running_stats=False)
    elif type=='bn':
      self.bn = nn.BatchNorm2d(channels)
    elif type=='group':
      self.bn = nn.GroupNorm(n_groups, mid_channels, affine=False),
    else:
      self.bn = None

  def forward(self, x):
    if self.bn is not None: 
      return self.bn(x)
    else:
      return x

# Divisive normalization activation function
class DivisiveNorm(nn.Module):
  def __init__(self, nr_filters=None):
    super().__init__()
    #self.alpha = torch.nn.Parameter(torch.tensor([1.0]))
    self.alpha = 2
    self.beta = torch.nn.Parameter(torch.tensor([0.1]))
    self.gamma = torch.nn.Parameter(torch.ones(nr_filters))

  def forward(self, x):
    # x has shape NCHW or NCTHW
    #x_a = x.pow(self.alpha)
    #y = self.gamma * x_a / (self.beta.pow(self.alpha) + torch.sum(x_a, dim=1, keepdim=True))

    xs = x.size()
    gs = [1]*len(xs)
    gs[1] = xs[1]

    y = x / (self.beta + torch.sum(self.gamma.reshape(gs)*x*x, dim=1, keepdim=True)).sqrt()
    return y

# activation function to replace the defaul activation
class Activation(nn.Module):
  def __init__(self, nr_filters=None):
    super().__init__()

    self.act = nn.ReLU()
    #self.act = nn.ELU()
    #self.act = nn.LeakyReLU()
    #self.act = nn.PReLU()
    #self.act = nn.ReLU6()
    #self.act = nn.RReLU()
    #self.act = nn.SELU()
    #self.act = nn.CELU()
    #self.act = DivisiveNorm(nr_filters)
    #self.act = GDN(nr_filters)


  def forward(self, x):
    return self.act(x)

# a basic convolution block
class ConvBlock(nn.Module):
  def __init__(self, \
      in_channels, 
      out_channels,
      stride = 1,
      dropout=False,
      bn = False, #batch norm
      activation=True):
    super().__init__()

    self.block = nn.ModuleList()
    self.block.append(IntConv2d(in_channels, 
        out_channels,
        kernel_size=3,
        stride = stride,
        padding=1))

    if activation:
      if bn: self.block.append(BN(out_channels))
      self.block.append(Activation(out_channels))

    # hardcoded dropout ratio for conv layer
    if dropout: 
      self.block.append(nn.Dropout(p=0.5))
    else: 
      self.block.append(nn.Sequential())

  def forward(self, x):
    for m1 in self.block:
      x = m1(x)
    return x

# a basic convolution block
class ConvTransposeBlock(nn.Module):
  def __init__(self, \
      in_channels, 
      out_channels,
      stride = 2,
      dropout=False,
      bn = False, #batch normalization
      activation=True):
    super().__init__()

    self.block = nn.ModuleList()
    self.block.append(nn.ConvTranspose2d(in_channels, 
        out_channels, 
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1))

    if activation:
      if bn: self.block.append(BN(out_channels))
      self.block.append(Activation(out_channels))

    if dropout: 
      self.block.append(nn.Dropout(p=0.5))
    else:
      self.block.append(nn.Sequential())

  def forward(self, x):
    for m1 in self.block:
      x = m1(x)
    return x
  
# Conv2d plus pixelshuffle to replace TransposeConv
class PSConv2d(nn.Module):
  def __init__(self, 
      in_channels,
      out_channels, 
      r=2):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conv = IntConv2d(in_channels, out_channels*(r**2), 
      kernel_size=3, padding=1)
    self.ps = nn.PixelShuffle(r)

  def forward(self, x):
    x = self.conv(x)
    x = self.ps(x)
    return x

# NN upsampling plus Conv2d to replace TransposeConv
class NNConv2d(nn.Module):
  def __init__(self, 
      in_channels,
      out_channels, 
      r=2):
    super().__init__()
    self.conv1x1 = IntConv2d(in_channels, out_channels*r*r, kernel_size=1)
    self.nn = nn.UpsamplingNearest2d(scale_factor=r)
    self.conv = IntConv2d(out_channels*r*r, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.conv1x1(x)
    x = self.nn(x)
    x = self.conv(x)
    return x


# if stride is negative, PSConv2d is used to increase the size of feature map
def get_shortcut(in_channels, out_channels, stride):
  if stride != 1:
    if stride > 0:
      shortcut = IntConv2d(in_channels, out_channels, 
        kernel_size=3, padding=1, stride=stride)
    else:
      shortcut = PSConv2d(in_channels, out_channels, 
        r=-stride)
  elif in_channels!=out_channels:
    shortcut = IntConv2d(in_channels, out_channels, 
      kernel_size=1)
  else:
    shortcut = nn.Identity() 
  return shortcut


# Not this implementation has activation before the shortcut, 
# Official ResBlock has activation after the shortcut
class ResBlock(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      stride = 1
      ): 
    super().__init__()

    self.conv1 = IntConv2d(in_channels, out_channels, 
    kernel_size=3, stride=stride, padding=1)

    self.act1 = nn.PReLU()
    self.conv2 = IntConv2d(out_channels, out_channels, 
      kernel_size=3, padding=1)
    self.act2 = nn.PReLU()
    self.shortcut = get_shortcut(in_channels, out_channels, stride)

  def forward(self, x):
    shortcut = self.shortcut(x)
    x = self.conv1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.act2(x)
    return x + shortcut

class ResBlockUp(ResBlock):
  def __init__(self, in_channels, out_channels, up_module):
    super().__init__(in_channels=in_channels, out_channels=out_channels, stride=1)
    self.conv1 = up_module(in_channels, out_channels)
    self.shortcut = get_shortcut(in_channels, out_channels, -2)

# A chain of resnet blocks
class ResBlocks(nn.Module):
  def __init__(self,
      in_channels, 
      num_blocks=5,
      block_type='basic', #'basic | bottleneck | resnext | gated
      skip_connection='conv2d', # conv2d | indentity
      dropout=False):
    super().__init__()  

    self.blocks = nn.ModuleList()

    if block_type=='basic':
      blk = IntBasicBlock(in_channels, in_channels, norm_layer=BN)

    else: 
      raise NotImplementedError(f"block type {block_type} is not supported")

    # Build Resblocks based on 1-block and 2-block ResBlocks elements. E.g. num_blocks=7 => 2+2+2+1
    if num_blocks <= 2:
      for ii in range(num_blocks):
        self.blocks.append(blk)
        if dropout: 
          self.blocks.append(nn.Dropout(p=0.5))
        else: 
          self.blocks.append(nn.Sequential())
    else:
      for ii in range(num_blocks//2):
        self.blocks.append(ResBlocks(in_channels, 2, block_type, skip_connection, dropout))
      #for ii in range(2):
      #  self.blocks.append(ResBlocks(in_channels, num_blocks//2, block_type, skip_connection, dropout))

      if num_blocks % 2 == 1:
        self.blocks.append(blk)
        if dropout: 
          self.blocks.append(nn.Dropout(p=0.5))
        else: 
          self.blocks.append(nn.Sequential())

    # skip
    self.skip = nn.Sequential(
        IntConv2d(in_channels, in_channels, kernel_size=1),
        #BN(in_channels),
        #Activation(in_channels)
        ) if skip_connection == "conv2d" else nn.Identity()

    # output fuse component
    self.output = nn.Sequential(
      BN(in_channels),
      Activation(in_channels),
      #nn.Dropout()
      )

  def forward(self, x):
    x1 = self.skip(x)
    for m1 in self.blocks:
      x = m1(x)
    return self.output(x+x1)
