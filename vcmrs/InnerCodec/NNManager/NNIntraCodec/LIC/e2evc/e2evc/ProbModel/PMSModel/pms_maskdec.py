# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .pms_utils import *
from e2evc.Utils.utils import *

from . import entropy_models
#from .chnl_dep import *


##########################################
# From Balle's tensorflow compression examples
#SCALES_MIN = 0.11
#SCALES_MAX = 256
#SCALES_LEVELS = 64

# torch.exp(torch.linspace(np.log(min), np.log(max), levels))
scale_table=[1.1000e-01, 1.2440e-01, 1.4069e-01, 1.5912e-01, 1.7995e-01, 2.0352e-01,
        2.3017e-01, 2.6031e-01, 2.9439e-01, 3.3294e-01, 3.7654e-01, 4.2585e-01,
        4.8161e-01, 5.4468e-01, 6.1600e-01, 6.9666e-01, 7.8789e-01, 8.9106e-01,
        1.0077e+00, 1.1397e+00, 1.2889e+00, 1.4577e+00, 1.6486e+00, 1.8645e+00,
        2.1086e+00, 2.3848e+00, 2.6970e+00, 3.0502e+00, 3.4496e+00, 3.9013e+00,
        4.4122e+00, 4.9899e+00, 5.6434e+00, 6.3823e+00, 7.2181e+00, 8.1633e+00,
        9.2322e+00, 1.0441e+01, 1.1808e+01, 1.3355e+01, 1.5103e+01, 1.7081e+01,
        1.9318e+01, 2.1847e+01, 2.4708e+01, 2.7944e+01, 3.1603e+01, 3.5741e+01,
        4.0421e+01, 4.5714e+01, 5.1700e+01, 5.8470e+01, 6.6127e+01, 7.4786e+01,
        8.4579e+01, 9.5654e+01, 1.0818e+02, 1.2235e+02, 1.3837e+02, 1.5648e+02,
        1.7698e+02, 2.0015e+02, 2.2636e+02, 2.5600e+02]

# get scale table, used by entropy model
def get_scale_table():
  return torch.tensor(scale_table)


class PMSMaskedDecoder(nn.Module):
    def __init__(self,
        out_channels=3,
        profile=None,
        ):
        '''
            Parameters
            ----------
                out_channels: output channels, for image compression, this is 3
                profile.block_size: the block size to determine the decoding order
                profile.mask_type: the type of encoding/decoding order, can be "default", "random", "entropy_map" and "inverse_entropy_map"
                profile.nr_filters: number of filters of the main branch
                profile.nr_resblocks: number of resnet blocks in the CNN module
                profile.att_key_dim: attention module key dimension, 0 indicate no attention module
                profile.att_val_dim: attention moduel value dimension 
                profile.quant_train: quantizaiton training strategy, 'noise', 'ste'
        '''
        super().__init__()
        self.profile = profile
        self.block_size = profile.block_size 

        # fix
        if profile.nr_mixtures == -1:
          # gaussian conditional model
          # get number of output channels
          # for MeanScaleHyperPrior model, the output of statistical parameters are
          # means and scales
          mix_channels = 2 * out_channels
  
          # entropy model
          # set scale table to be None
          self.em = entropy_models.GaussianConditional(None, quant_train=profile.quant_train)
        else:
          # gaussian mixture model
          # K(nr_mixtures) of mean, scale and weights for each elements
          mix_channels = 3 * profile.nr_mixtures * out_channels
  
          # entropy model
          # set scale table to be None
          self.em = entropy_models.GaussianMixConditional(None, quant_train=profile.quant_train)
 
        # channel dependency module, in 'none' mode, this module does nothing
        #self.chnl_deps = ChannelDepsModule(out_channels, 
        #  channel_mode=profile.channel_mode, 
        #  lead_channels=profile.lead_channels)

 
        # masked cnn, backbone network
        self.maskedcnn = PMSMaskedCNNB(
            y_input_channels=out_channels, 
            mix_channels=mix_channels, 
            profile=profile,
            )
        
        # upsampler
        self.y_updampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.z_upsampler = Upsampler(profile.nr_filters, scale=2)

    def forward(self, z2, x1, x0, kv_buffer=None):
        '''
            Parameters
            ----------
                z2 : tensor in NCHW format, output from the previous scale
                x_1 : tensor in NCHW format, the low resolution image
                x0 : tensor in NC(2H)(2W) format, ground truth of hign resolution image
                kv_buffer : key-value buffer for attenion module
            Return
            ------
                z : output context
                y : output tensor after qunatization
                entropy : entropy 
        '''
        # upsample
        y = pad_if_needed(self.y_updampler(x1), x0.size())
        if torch.is_tensor(z2): # the last scale has no z
          z = pad_if_needed(self.z_upsampler(z2), x0.size())
        else:
          z = z2

        # initial mask and entropy
        mask = self._get_init_mask(y)
        entropy = torch.Tensor([0.]).to(x0.device)

        step = 0
        while mask.sum() < mask.numel():
          z, mix = self.maskedcnn(z, y, mask, kv_buffer) 

          if self.profile.nr_mixtures == -1:
            means, scales = torch.chunk(mix, 2, dim=1)

            # apply channel dependency module
            # differnt type of channel dependency module
            # y_bar, means = self.chnl_deps(self.em, x0, means)

            # get entropy map
            y_bar, true_entropy_map = self.em(x0, scales, means)
          else:
            means, scales, weights = torch.chunk(mix, 3, dim=1)

            # apply channel dependency module
            # differnt type of channel dependency module
            # y_bar, means = self.chnl_deps(self.em, x0, means)

            # get entropy map
            y_bar, true_entropy_map = self.em(x0, scales, means, weights)

          true_entropy_map = true_entropy_map * (1-mask) # mark already decoded pixels to 0

          # update mask according to the mixture output
          prev_mask = mask
          mask = self._update_mask(y, mix, mask, step)

          # update entropy and ground truth according to the new mask
          entropy += (true_entropy_map * mask).sum() 
          y = y_bar*(mask-prev_mask) + y*(1+prev_mask-mask)

          step += 1

        return z, y, entropy

    def update(self):
        ''' Update CDFs before compress/decompress
        '''
        scale_table = get_scale_table()
        self.em.update_scale_table(scale_table, force=False)

    def compress(self, bitstream, z2, x1, x0, encoding_mask, kv_buffer=None):
        '''
            Parameters
            ----------
                bitstream: bitstream that new bits are added
                z2 : tensor in NCHW format, output from the previous scale
                x1 : tensor in NCHW format, the low resolution image
                x0 : tensor in NC(2H)(2W) format, ground truth of hign resolution image
                encoding_mask : NC(2H)(2W), not used? 
                kv_buffer: key-value buffer for attention module
            Return
            ------
                z : output context
        '''
        y = pad_if_needed(self.y_updampler(x1), x0.size())
        if torch.is_tensor(z2): # the last scale has no z
          z = pad_if_needed(self.z_upsampler(z2), x0.size())
        else:
          z = z2

        mask = self._get_init_mask(y)
        entropy = torch.Tensor([0.]).to(x0.device)

        step = 0
        while mask.sum() < mask.numel():
          z, mix = self.maskedcnn(z, y, mask, kv_buffer) 
          means, scales = torch.chunk(mix, 2, dim=1)

          # apply channel dependency module
          # different type of channle dependency module
          # y_bar, means = self.chnl_deps(self.em, x0, means)

          y_bar, true_entropy_map = self.em(x0, scales, means)

          true_entropy_map = true_entropy_map * (1-mask) # mark already decoded pixels to 0

          # update mask according to the mixture output
          prev_mask = mask
          mask = self._update_mask(y, mix, mask, step)

          # compress 

          N, C, H, W = x0.size()
          symbols_mask = (mask - prev_mask)==1 # NCHW
          if symbols_mask.size(1)==1: symbols_mask = symbols_mask.repeat(1,C,1,1) #NCHW

          symbols_size = [N, 
            symbols_mask.sum(dim=1).max().item(), 
            symbols_mask.sum(dim=2).max().item(),
            symbols_mask.sum(dim=3).max().item()]

          assert symbols_mask.sum() == np.prod(symbols_size), \
            'Only one pixel per block at one step is supported'

          symbols_means = means[symbols_mask].view(symbols_size)
          symbols_scales = scales[symbols_mask].view(symbols_size)
          symbols = x0[symbols_mask].view(symbols_size)
          indexes = self.em.build_indexes(symbols_scales)

          # compress
          self.em.compress(bitstream, symbols, indexes, means=symbols_means)

          # update entropy and ground truth according to the new mask
          entropy += (true_entropy_map * mask).sum() 
          y = y_bar*(mask-prev_mask) + y*(1+prev_mask-mask)

          step += 1

        return z

    def decompress(self, bitstream, z2, x1, encoding_mask, pre_x0, kv_buffer=None):
        '''
            Parameters
            ----------
                bitstream: bitstream to be decompressed
                z2 : tensor in NCHW format, output from the previous scale
                x1 : tensor in NCHW format, the low resolution image
                encoding_mask: a boolean mask indicate which symbols was available in the bitstream
                pre_x0 : pre-filled x0 value that matches the encoding mask
                kv_buffer: key-value buffer for attention module
            Return
            ------
                z : output context
                y : output tensor after dequnatization
        '''

        y = pad_if_needed(self.y_updampler(x1), pre_x0.size())
        # fill prefilled x0
        pre_mask=~encoding_mask.repeat(1, y.size(1),1,1)
        if torch.is_tensor(z2): # the last scale has no z
            z = pad_if_needed(self.z_upsampler(z2), pre_x0.size())
        else:
            z = z2

        mask = self._get_init_mask(y)

        step = 0
        while mask.sum() < mask.numel():
            z, mix = self.maskedcnn(z, y, mask, kv_buffer)
            means, scales = torch.chunk(mix, 2, dim=1)

            prev_mask = mask

            # update mask according to the mixture output
            mask = self._update_mask(y, mix, mask, step)

            # get mask for symbols to be decoded
            N, C, H, W = y.size()

            symbols_mask = (mask - prev_mask)==1 # NCHW
            if symbols_mask.size(1)==1: symbols_mask = symbols_mask.repeat(1,C,1,1) #NCHW

            symbols_size = [N, 
              symbols_mask.sum(dim=1).max().item(), 
              symbols_mask.sum(dim=2).max().item(),
              symbols_mask.sum(dim=3).max().item()]
            assert symbols_mask.sum() == np.prod(symbols_size), \
              'Only one pixel per block at one step is supported'

            symbols_means = means[symbols_mask].view(symbols_size)
            symbols_scales = scales[symbols_mask].view(symbols_size)

            # using channel dependency module to docode
            # another implementation of channel dependencies
            # values = self.chnl_deps.decompress(self.em, bitstream, symbols_means, symbols_scales)

            indexes = self.em.build_indexes(symbols_scales)
            values = self.em.decompress(bitstream, indexes, symbols_means)

            # set values
            y[symbols_mask] = values.view(-1)

            step += 1

        return z, y
    
    def _update_mask(self, y, mix, mask, step):
        ''' update mask according to new mix output
            with channel dependencies
            Parameters
            ----------
                y : current decoded tensor
                mix : mixture probability model parameters
                mas : current mask
                step : current step
            Return
            ------
                new mask in shape N C H W
        '''
        N, C, H, W = y.size()
        #new_mask = torch.zeros_like(mask, requires_grad=False)
        new_mask = mask.detach().clone()
        spatial_step = step

        if self.profile.channel_mode =='mixture':
            spatial_step = step // (self.profile.mixture_seeds+1)
            chnl_step = step % (self.profile.mixture_seeds+1)
            if chnl_step < self.profile.mixture_seeds:
              chnl_rng = (chnl_step, chnl_step+1)
            else:
              chnl_rng = (chnl_step, C)
            param_chnl_rng = chnl_rng
           
        else: 
            raise NotImplementedError

        if self.profile.mask_type == 'default':
            # predefined order
            # get predefied order key
            order = self.profile.predefined_pixel_order[ \
              self.block_size[0]*1000 + self.block_size[1]][spatial_step]
            new_mask[:,chnl_rng[0]:chnl_rng[1],order[0]::self.block_size[0],order[1]::self.block_size[1]] = 1

            return new_mask
        
        else:
            assert False, f'Mask type {self.mask_type} not supported!'

        entropy_map += -mask * 1E20 #make already decoded pixels to infinite small number
        _, max_indices = F.max_pool2d(entropy_map,kernel_size=self.block_size, return_indices=True)
        pos_mask = F.max_unpool2d(torch.ones_like(max_indices), max_indices, kernel_size=self.block_size) # N1HW
        new_mask = pos_mask.repeat(1,mask.size(1),1,1) # NCHW

        mask = mask + new_mask 
        return mask


    def _get_init_mask(self, y):
        ''' get initial mask
        '''
        mask_size = list(y.size()) # NCHW
        if self.profile.channel_mode == 'none':
          mask_size[1] = 1
        mask = torch.zeros(mask_size, requires_grad=False, device=y.device)
        mask[:, :, 1::2, 1::2] = 1
        return mask


class PMSMaskedCNNB(nn.Module):
    def __init__(self,
        y_input_channels,
        mix_channels,
        profile,
        ):
        '''
        Parameters
        ----------
            y_input_channels: input channels of y. For image compression, this is 3. 
            mix_channels: number of output channels for mixture probability
            nr_filters: the size of the main branch
            nr_resblocks: number of resblocks of the main branch
            att_v_dim: attention module value dimension, if 0, no attention module
        '''
        super().__init__()
        nr_filters = profile.nr_filters
        self.profile = profile
        self.nr_filters = nr_filters

        # mask is the same size as y
        if profile.channel_mode == 'none':
          conv1_in_channels = y_input_channels+1+nr_filters # mask with sizes 1
        else:
          conv1_in_channels = y_input_channels*2+nr_filters # mask with size C

        self.conv1 = ConvBlock(conv1_in_channels, nr_filters) 

        self.out_z_conv = BasicConv(nr_filters, nr_filters)
        self.out_mix_conv = BasicConv(nr_filters, mix_channels)

        self.resblocks = nn.Sequential()
        for ii in range(profile.nr_resblocks):
            self.resblocks.add_module(f"res{ii}", ResBlockA(nr_filters, nr_filters))

    def forward(self, x, y, mask, kv_buffer=None):
        # x: corresponding the z tensor, the output from the previous maskedcnn
        # y: input (upsampled from low dim image)
        # mask: mask
        # kv_buffer: key/value buffer for attension module

        # with sum, it doesn't matter if x is a scalar or a tensor. In here we have to create a zero tensor

        if not torch.is_tensor(x):
            x_shape = list(mask.size())
            x_shape[1] = self.nr_filters
            x = torch.zeros(x_shape, dtype=mask.dtype, device=mask.device)

        x = torch.cat([y, mask, x], dim=1)
        x = self.conv1(x)

        x = self.resblocks(x)

        x = self.out_z_conv(x)
        mix_out = self.out_mix_conv(x)
        return x, mix_out

