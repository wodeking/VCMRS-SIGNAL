# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import warnings
import torch
from torch import nn
from torchvision.models import resnet
import copy

from e2evc.Utils.utils import *
from e2evc.EntropyCodec import *
from e2evc.Utils import ctx
# from e2evc.Utils.dbg_tools import *

from .pms_maskdec import *
from .entropy_models import *

from . import profiles

#
# Progressive Multi-scale model: one MaskCNN for every scale
#
class PMSModel(nn.Module):
    def __init__(self,
                 input_channels,
                 profile='normal',
                 ):
        '''
        Parameters
        ----------
            input_channels: number of input channels
            profile : Profile name
        '''
        super().__init__()

        # get profile
        if isinstance(profile, str):
            profile = getattr(profiles, profile)

        self.profile = profile
        self.scales = profile.scales

        # downscale encoder
        self.encoder = FixedPooling(scale=2)

        # masked decoder
        self.prob_decoder = PMSMaskedDecoder(
            out_channels = input_channels,
            profile = profile,
            )

        # entropy model for the last scale
        self.last_em = EntropyBottleneck(input_channels, \
          quant_train=profile.quant_train)

        # attention kv buffer
        self.kv_buffer = []

    def get_downsampled(self, x_hat):
        ''' get downsampled images for each scale
        '''
        downsampled = []
        x = x_hat

        for ii in range(self.scales):
            x, x_, round_bits = self.encoder(x)
            downsampled.append([x_, x, round_bits])
        return downsampled

    def get_downsampled_mask(self, x_hat, mask):
        ''' get downsampled mask
        '''
        downsampled = []
        if mask is None: mask = torch.ones_like(x_hat)[:,[0],:,:]
        mask = mask.to(bool)
        downsampled.append(mask)
        for ii in range(self.scales):
          mask = FixedPooling(scale=2)(mask)[0]
          downsampled.append(mask)
        return downsampled


    def forward(self, x_hat):
        '''
          Parameters
          ----------
            x_hat : tensor in NCHW, in float
        '''

        # check input, if it's integer, give a warning
        if torch.norm(torch.round(x_hat)-x_hat, 2)<1E-8:
          warnings.warn('Input to PMSModel is integer, float number are expected')
 
        rate_loss = torch.Tensor([0.]).to(x_hat.device)

        # get downsampled inputs
        downsampled = self.get_downsampled(x_hat)

        # the last scale is send via entropy bottleneck
        last_x, likelihood = self.last_em(downsampled[-1][0])
        downsampled[-1][0] = last_x
        downsampled[-1][1] = last_x
        rate_loss += likelihood.sum() 

        #auxiliary loss from EntropyBottleneck, to learn the range of input data
        # this is not entropy. However for simplicity, this is added to entropy
        # when it converges, the value should be very small
        aux_loss = 0.001 * self.last_em.loss()
        rate_loss += aux_loss

        # prepare the input from the last scale
        x1 = last_x
        z2 = 0.0

        # attention buffer
        kv_buffer = []

        # train and compress in reverse order
        for ii in reversed(range(self.scales)):

            x0 = downsampled[ii-1][1] if ii > 0 else x_hat

            z2, y_hat, entropy = self.prob_decoder(z2, x1, x0, kv_buffer)

            # calculating the entropy for this scale
            rate_loss += entropy

            x1 = y_hat

        return y_hat, rate_loss

   
    def update(self):
        ''' Update the CDFs before compress or decompress
        '''
        self.last_em.update()
        self.prob_decoder.update()
    
    # @ctx.int_conv()
    def compress(self, bitstream, x_hat, encoding_mask=None):
        ''' compress input tensor in 1CHW format and return the encoded bitstream
            Parameters
            ----------
              bitstream: the bitstream to which the symbols are encoded
              x_hat: input tensor
              encoding_mask: the mask indicate which symbols in x_hat to be encoded
        '''
        assert x_hat.size(0) == 1, 'Compressing more than one tensor is not supported'

        # check input, if it's in integer, give a warning
        if torch.norm(torch.round(x_hat)-x_hat, 2)<1E-8:
          warnings.warn('Input to PMSModel is integer, float number are expected')
 
        rate_loss = torch.Tensor([0.]).to(x_hat.device)

        # get downsampled inputs
        downsampled = self.get_downsampled(x_hat)
        downsampled_masks = self.get_downsampled_mask(x_hat, encoding_mask)

        # the last scale is send via last_em(entropy bottleneck)
        last_x, likelihood = self.last_em(downsampled[-1][0])
        self.last_em.compress(bitstream, downsampled[-1][0])
        downsampled[-1][0] = last_x
        downsampled[-1][1] = last_x
        rate_loss += likelihood.sum() 

        # prepare the input from the last scale
        x1 = last_x
        z2 = 0.0

        # attention buffer
        kv_buffer = []

        # train and compress in reverse order
        for ii in reversed(range(self.scales)):

            x0 = downsampled[ii-1][1] if ii > 0 else x_hat

            # debug: to tensors in different scales
            # dbg_save_feature('pms_scale_'+str(ii), x0)

            kv_buffer_copy = copy.copy(kv_buffer)
            
            z2_enc, y_hat, entropy = self.prob_decoder(z2, x1, x0, kv_buffer_copy)
            z2 = self.prob_decoder.compress(bitstream, z2, x1, x0, downsampled_masks[ii], kv_buffer)

            # calculating the entropy for this scale
            rate_loss += entropy

            x1 = y_hat

        return y_hat, rate_loss

    def decompress(self, bitstream, xs, encoding_mask=None, pre_x0=None):
        ''' decompress an tensor from bitstream

        Parameters
        ----------
          bitstream : bitstream
          xs : the shape of output tensor, in format N, C, H, W
          encoding_mask: the bits indicate the pixels that should be decoded 
          pre_x0: pre-filled x0, where some pixels are already pre-filled. The pre-filled data matches the negative encoding_mask

        Return
        ------
          tensor that has been recovered from the input bitstream
        '''
        #assert len(xs) == 4 and xs[0]==1, 'input shape should be the format of 1, C, H, W'

        N, C, H, W, = xs

        # get downsampled prefilled x0 and encoding mask 
        pre_x0 = torch.zeros(xs).to(next(self.parameters()).device) if pre_x0 is None else pre_x0
        downsampled = self.get_downsampled(pre_x0)
        pre_x0s = [pre_x0] 
        for px,_,_ in downsampled: pre_x0s.append(px)
        downsampled_masks = self.get_downsampled_mask(pre_x0, encoding_mask)

        # the last context
        z2 = 0  # the last context

        # decode the last downsample image using last_em(entropy_model)
        x1 = pre_x0s[-1].contiguous()
        last_x = self.last_em.decompress(bitstream, x1.size()[2:])

        enc_mask = downsampled_masks[-1]
        x1[enc_mask.repeat(1,C,1,1)] = last_x.view(-1)

        # attention buffer
        kv_buffer = []

        # train and compress in reverse order
        for ii in reversed(range(self.scales)):
            z2, x0 = self.prob_decoder.decompress(
              bitstream, 
              z2, 
              x1, 
              downsampled_masks[ii], 
              pre_x0s[ii], 
              kv_buffer)

            z2.detach()
            x0.detach()
            x1 = x0

        return x0
