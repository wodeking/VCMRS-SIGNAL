# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
# channel dependency module

import torch
from torch import nn
from ...Utils.utils_nn import IntConv2d

class ChannelDepsModule(nn.Module):
  '''Channel dependency module
  '''
  def __init__(self, mix_channels, channel_mode='none', lead_channels=4):
    '''
      mix_channels: number of mixture parameters
      channel_mode: 'none', 'previous', 'full', 'mixture'
      lead_channels: default 4
    '''
    super().__init__()

    self.mix_channels = mix_channels
    self.mode = channel_mode
    self.num_lead = lead_channels # number of lead channels in full dependent mode

    if self.mode == 'previous':
      #  Use a C-1 weight vector multiply x[:, 0:C-1, :, :] + mix
      self.weights = nn.Parameter(torch.zeros(self.mix_channels-1))

    elif self.mode == 'full':
      # full-dependent mode implemented by a sequence of 1x1 conv
      self.body = nn.ModuleList()
      for idx in range(self.mix_channels-1):
        self.body.append(
          IntConv2d(idx+1, 1, kernel_size=1, padding=0))

    elif self.mode == 'mixture':
      # lead is in full-dependent mode
      self.lead_module = nn.ModuleList()
      for idx in range(self.num_lead-1):
        self.lead_module.append(
          IntConv2d(idx+1, 1, kernel_size=1, padding=0))
 
      self.body = IntConv2d(self.num_lead, self.mix_channels - self.num_lead,
        kernel_size=1, padding=0)

    elif self.mode == 'none':
      pass

    else:
      assert False, f'chanel_dependency: {channel_mode} is not supported'


  def forward(self, em, x, mix):
    '''
    forward and compress perform the same operation
    Parameters:
      em: entropy model, implemented _quantize function
      x: ground truth data
      mix: probability model parameters
    returns:
      y_bar: quantized input
      mix_out: output of probability parameters
    '''
    N, C, H, W = x.shape
    quant_method = 'noise' if self.training else 'dequantize'

    if self.mode == 'previous':
      mix_out = [mix[:, [0], :, :]]
      y_bar = [em._quantize(x[:, [0], :, :], quant_method, mix[:, [0], :, :])]

      for idx in range(1, C):
        #self.weight: C-1
        mix_chnl = y_bar[idx-1]*self.weights[idx-1].unsqueeze(-1).unsqueeze(-1) + \
          mix[:,[idx],:,:]
        y_bar.append(em._quantize(x[:,[idx],:,:], quant_method, mix_chnl))
        mix_out.append(mix_chnl)

      y_bar = torch.cat(y_bar, dim=1)
      mix_out = torch.cat(mix_out, dim=1)

    elif self.mode == 'full':
      mix_out = [mix[:, [0], :, :]]
      y_bar = em._quantize(x[:, [0], :, :], quant_method, mix[:, [0], :, :])

      for idx in range(1, C):
        mix_chnl = self.body[idx-1](y_bar) + mix[:,[idx],:,:]
        y_bar = torch.cat([y_bar, em._quantize(x[:,[idx],:,:], quant_method, mix_chnl)], dim=1)
        mix_out.append(mix_chnl)

      mix_out = torch.cat(mix_out, dim=1)

    elif self.mode == 'mixture':
      # lead channels are in full-dependent mode
      mix_out = [mix[:, [0], :, :]]
      y_bar = em._quantize(x[:, [0], :, :], quant_method, mix[:, [0], :, :])
      for idx in range(1, self.num_lead):
        mix_chnl = self.lead_module[idx-1](y_bar) + mix[:,[idx],:,:]
        y_bar = torch.cat([y_bar, em._quantize(x[:,[idx],:,:], quant_method, mix_chnl)], dim=1)
        mix_out.append(mix_chnl)

      mix_chnl = self.body(y_bar) + mix[:, self.num_lead:, :, :]
      y_bar_chnl = em._quantize(x[:, self.num_lead:, :, :], quant_method, mix_chnl)
      mix_out.append(mix_chnl)
      mix_out = torch.cat(mix_out, dim=1)
      y_bar = torch.cat([y_bar, y_bar_chnl], dim=1)

    elif self.mode == 'none':
      y_bar = em._quantize(x, quant_method, mix) 
      mix_out = mix

    else:
      raise NotImplementedError
   
    return y_bar, mix_out

  def decompress(self, em, bitstream, means, scales):
    '''
    Parameters:
      em: entropy model, implement _quantize function
      bitstream: bitstream that can be used to decode symbols
      mix: probability model parameters
    returns:
      values: decoded values, same shape as means
    '''
    #values = self.chnl_deps.decompress(em, bitstream, symbol_means, symbols_scales)

    N, C, H, W = means.shape
    if self.mode == 'previous':
      # first channel
      indexes = em.build_indexes(scales[:,[0],:,:])
      chnl_values = em.decompress(bitstream, indexes, means[:,[0],:,:]) # N,1,H,W
      values = [chnl_values]

      for idx in range(1, C):
        #self.weight: C
        chnl_means = values[-1] * self.weights[idx-1].unsqueeze(-1).unsqueeze(-1) + \
          means[:,[idx],:,:] # N,1,H,W
        indexes = em.build_indexes(scales[:,[idx],:,:])
        chnl_values = em.decompress(bitstream, indexes, chnl_means) # N,1,H,W
        values.append(chnl_values)
      values = torch.cat(values, dim=1) # N,C,H,W

    elif self.mode == 'full':
      # first channel
      indexes = em.build_indexes(scales[:,[0],:,:])
      values = em.decompress(bitstream, indexes, means[:,[0],:,:]) # N,1,H,W

      for idx in range(1, C):
        chnl_means = self.body[idx-1](values) + means[:,[idx],:,:] #N,1,H,W
        indexes = em.build_indexes(scales[:,[idx],:,:])
        chnl_values = em.decompress(bitstream, indexes, chnl_means) # N,1,H,W
        values = torch.cat([values, chnl_values])

    elif self.mode == 'mixture':
      # lead channels are in full-dependent mode
      indexes = em.build_indexes(scales[:,[0],:,:])
      values = em.decompress(bitstream, indexes, means[:,[0],:,:]) # N,1,H,W

      for idx in range(1, self.num_lead):
        chnl_means = self.lead_body[idx-1](values) + means[:,[idx],:,:] #N,1,H,W
        indexes = em.build_indexes(scales[:,[idx],:,:])
        chnl_values = em.decompress(bitstream, indexes, chnl_means) # N,1,H,W
        values = torch.cat([values, chnl_values], dim=1)

      # the other channels
      chnl_means = self.body(values) + means[:,self.num_lead:,:,:]
      indexes = em.build_indexes(scales[:,self.num_lead:,:,:])
      chnl_values = em.decompress(bitstream, indexes, chnl_means) # N,1,H,W
      values = torch.cat([values, chnl_values], dim=1)

    elif self.mode == 'none':
      indexes = em.build_indexes(scales)
      values = em.decompress(bitstream, indexes, symbols_means)

    else:
      raise NotImplementedError
   
    return values


