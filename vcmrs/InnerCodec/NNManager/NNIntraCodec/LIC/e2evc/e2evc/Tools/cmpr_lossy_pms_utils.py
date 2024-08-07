# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

# compress and decompress tool using a lossy engine with an entropy model 

import time
import os
import pprint
import numpy as np
import torch

from io import BytesIO
from e2evc.Tools.cmpr_utils import *

from e2evc.Utils import utils as e2eutils
from e2evc.EntropyCodec import RansEntropyCoder


def compress(model, x, o_fname):
  '''compress x_hat a lossy model
     x must be float in range [-1, 1]
  '''

  xs = x.size()

  entropy_encoder = RansEntropyCoder()

  # start_time = time.time()
  # model.cuda()
  with torch.no_grad():
    # we need to update CDF for entropy model first
    model.entropy_model.update()

  model.compress(entropy_encoder, x)
  bitstream = entropy_encoder.flush()
  
  with open(o_fname, 'wb') as f:
    # NOTE: Image size will not be written to the bitstream anymore
    # headers contains the size of the compressed tensor and scales
    # f.write(np.array([W, H, C], dtype=np.uint16).tobytes('C'))
    f.write(bitstream)
  
  total_fsize = os.path.getsize(o_fname)
  cal_bpp = total_fsize * 8 / np.prod(xs) * xs[1]
  return cal_bpp

def decompress(model, fname, original_size, full_range=False):
  ''' decompress a file and return the data in format 1CHW
  ''' 
  # start_time = time.time()
  C, H, W = original_size
  # load headers and bitstreams
  # the file name may be in format <fname:start_byte>
  start_idx = 0
  if ':' in fname: fname, start_idx = fname.split(':') 
  start_idx = int(start_idx)
  with open(fname, 'rb') as f:
    # W,H,C = read_value_from_f(f, dtype=np.uint16, num=3)
    # read strings
    f.seek(start_idx)
    bitstream = f.read()

  entropy_decoder = RansEntropyCoder()
  entropy_decoder.set_stream(bitstream)
  
  N=1
  xs = [N, C, H, W]
  levels = 256

  # print('decompressing ....')

  # model.cuda()

  with torch.no_grad():
    model.eval()
    model.entropy_model.update()

    x0 = model.decompress(entropy_decoder, xs)
    x0 = e2eutils.float_round_int(x0, full_range_input=full_range)

  # print('Elapse: {}'.format(time.time()-start_time))
  return x0
  
