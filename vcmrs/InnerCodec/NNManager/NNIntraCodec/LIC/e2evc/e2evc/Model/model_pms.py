# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import time
from types import SimpleNamespace
import torch
from torch import nn
from e2evc.Datasets.transform_utils import get_padded_size
from e2evc.Utils.utils import getattr_by_path
from ..AutoEncoder import *
from ..ProbModel import *
from e2evc.Utils import ctx
import vcmrs
class E2EModelPMS(nn.Module):
  '''End-to-end model for lossy image compression with entropy model
     
  '''

  def __init__(self, opt):
    '''
    Parameters
    --------------
    opt:
      opt.lattent_dim: latent dimension
      opt.prob_model: probability(entorpy) model, should be one of _supported_entropy_models
    '''
    super().__init__()
    self.latent_dim = opt.latent_dim

    pms_profile = opt.probmodel.pms_profile

    self.entropy_model = PMSModel(input_channels=opt.latent_dim, profile=pms_profile)

    self.encoder = EncoderRes(input_channels=3, output_channels=self.latent_dim)
    self.decoder = DecoderRes(input_channels=self.latent_dim, output_channels=3)

  def forward(self, *args, **kwargs):
    raise AssertionError("The forward function shoud never be called")

  def em_decode(self, y):
    y_hat, pz = self.entropy_model(y)
    x_hat = self.decoder(y_hat)
    x_hat = torch.clamp(x_hat, -1, 1)
    return x_hat, y_hat, pz #pz is ce

  @torch.no_grad()
  @ctx.int_conv()
  def compress(self, bitstream, x):
    '''compress image
    '''
    # input should be float in [-1, 1]
    assert x.max()<=1 and x.min()>=-1, 'Input must be in range -1, 1'
    self.eval()
    
    # compress
    y = self.encoder(x)
    # using integer convolution
    with ctx.int_conv():
      self.entropy_model.compress(bitstream, y)

    return None
  
  @torch.no_grad()
  @ctx.int_conv()
  def decompress(self, bitstream, xs):
    '''decompress image
    Parameters
    ----------
      bitstream : input bitstream
      xs : Output image shape, in the form of NCHW
    '''
    self.eval()
    N, C, H, W = get_padded_size(xs, self.size_divisible)

    xs_pmodel = (N,self.latent_dim, H//8, W//8) # hacking, this should come from encoder

    # decompress using integer conv 
    y_hat = self.entropy_model.decompress(bitstream, xs_pmodel)
    #y_hat = y_hat.to(next(self.parameters()).device)
    y_hat = y_hat.view(xs_pmodel)

    #with ctx.int_conv(24):
    #  x_hat = self.decoder(y_hat)
    # decoder = self.decoder
    x_hat = self.decoder(y_hat)

    x_hat = torch.clamp(x_hat, -1, 1)
    return x_hat
  
  @staticmethod
  def get_log_data_types(stage, input, outputs):
    if stage in ['train', 'test', 'full']:
      x_hat, z, pz = outputs
      return (input, 'input', 'image'), (x_hat, 'output', 'image')
    else:
      assert False, f'Other stage is not supported: {stage}'



