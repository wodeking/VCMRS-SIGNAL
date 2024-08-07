# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
# utilities

from contextlib import nullcontext
from os import path
import random
from warnings import warn
from PIL import Image

import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

import cv2
import torch
import torchvision
import numpy as np
import pprint
import math
import logging
from collections import defaultdict
from torchvision.transforms.transforms import Compose, ToTensor
import vcmrs
# import nvidia_smi

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############################################
# system utilities
def fix_random_seed(seed=0):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def make_deterministic(is_deterministic=True):
  if is_deterministic:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #CUBLAS_WORKSPACE_CONFIG=:16:8
  else:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]

###############################################
class DummyObject:
  """
    A dummy object class that has its instances repsond to any function/attribute call doing nothing, instead of giving exception errors.
  """
  def __init__(self, *args, **kwargs):
    pass
  def __call__(self, *args, **kwargs):
    return self
  def __getattr__(self, *args, **kwargs):
    return self
###############################################
# visulization tools for testing

def show_img(x):
  '''show a image tensor on screen
  '''
  if x.dtype == torch.float:
    # input is between -1 and +1
    x = torch.round((x / 2. + 0.5 )*255)
    x = x.type(torch.uint8)
  elif x.dtype == torch.int:
    x = x.type(torch.uint8)
  x = x.cpu().numpy()
  plt.figure()
  plt.imshow(np.transpose(x, (1,2,0)), interpolation='nearest')
  plt.show()

def show_clip(clip):
  '''show a video clip tensor as grid images on screen

     clip with shape C,T,H,W
  '''
  clip = clip.cpu().permute(1,0,2,3)
  img_grid = torchvision.utils.make_grid(clip, nrow=8, normalize=True)
  show_img(img_grid)


###############################################
#


def load_img_tensor(path, full_range=False):
  """
    Reads the image at `path` into a Pytorch's mini-batch tensor (1xCxHxW) of value range [0, 1].
    If `full_range` is True, scales the values to [-1, 1].
  """
  img = Image.open(path)
  if img.mode!='RGB':
    img = img.convert('RGB')
  img = ToTensor()(img)
  if full_range:
    img = (img-0.5) * 2 # [0,1] to [-1, 1]
  return img[None] # unsqueeze

def cvt_yuv444p_to_yuv420p10b(in_yuv_data):
  ''' Convert image from yuv444p to yuv420p 10bit format
  
  Params
  ------
    yuv_data: in yuv444p format, 10 bit, WH3 
  '''
  yuv_data = in_yuv_data.transpose(2,1, 0) #3HW
  yuv_data = yuv_data.astype('uint16')

  if in_yuv_data.dtype == 'uint8':
    yuv_data *= 4

  yy,uu,vv = yuv_data
  H,W = yy.shape
  uu = uu.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3)).astype('uint16')
  vv = vv.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3)).astype('uint16')
  return yy, uu, vv

def to_limited_range(yuv_data):
  '''convert full range to limited range. Y: 16:235, U,V: 16:240
  '''
  tmp_data = yuv_data.astype(float)
  tmp_data[:,:,0] = tmp_data[:,:,0]/255 * (235-16) + 16
  tmp_data[:,:,[1,2]] = tmp_data[:,:,[1,2]]/255 * (240-16) + 16
  return tmp_data.astype('uint8')
 
def save_rgb2yuv420p_10b_precise(image, fname):
  ''' Convert image from RGB to yuv420p 10bit format accurately

  Params
  ------
    image: torch.float32 tensor, CHW, [0, 1]
  '''
  x = image.cpu().numpy()
  
  # use opencv for color conversion
  rgb = (x.transpose(1,2,0) * 255).astype('uint8')
  bgr = rgb[:,:,[2,1,0]]

  ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
  #ycrcb = to_limited_range(ycrcb)
  yy,vv,uu = ycrcb.transpose(2,0,1) / 255

  #r, g, b = x[0, :, :], x[1, :, :], x[2, :, :]
  #yy = 0.299*r + 0.587*g + 0.114*b
  #uu = (b - yy)*0.492 + 0.5
  #vv = (r - yy)*0.877 + 0.5

  H,W = yy.shape
  uu = uu.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))
  vv = vv.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))

  yy=np.clip((yy*1023).astype('uint16'), 0, 1023)
  uu=np.clip((uu*1023).astype('uint16'), 0, 1023)
  vv=np.clip((vv*1023).astype('uint16'), 0, 1023)

  with open(fname, 'wb') as of:
    of.write(yy.tobytes())
    of.write(uu.tobytes())
    of.write(vv.tobytes())

def save_img(x, fname, as_format='rgb'):
  '''save an image to a file

     Parameters
     -----------
     x : input tensor of shape W H 3. Data type: uint8 
        Color space: rgb
     fname: file name

  '''
  assert x.dtype in [torch.int, torch.uint8], f'data type not supported'

  if x.dtype == torch.int:
    x = x.type(torch.uint8)
    
  x = x.cpu().numpy()
  if as_format == 'rgb':
    mpimg.imsave(fname, x)
  elif as_format == 'yuv420p_10b':
    yuv444 = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    #convert YUV444 to yuv420 10p
    img_y, img_u, img_v = cvt_yuv444p_to_yuv420p10b(yuv444)
    with open(fname, 'wb') as of:
      of.write(img_y.tobytes())
      of.write(img_u.tobytes())
      of.write(img_v.tobytes())
 
  else:
    assert False, f"as_format: {as_format} not supported"

###############################################
#
def validate_input_int(x, levels=256):
    """ validate input range in x should be int in float32 type
    """
    assert x.dtype==torch.float32 and torch.norm(torch.round(x) - x,2)<1E-8, "Input should be int in float32 type"
    assert torch.min(x)>=0 and torch.max(x)<levels, f"Input should be between 0 and {levels-1}"
    return x


def int_to_float(z, levels=256):
  '''normalize from int to float in range -1, 1
  '''
  assert torch.min(z)>=0 and torch.max(z)<levels, f"Input is not in the range of 0 and {levels}"
  #x = (z.float()/(levels-1)-0.5)*2
  x = (z.float()*2 + 1)/levels - 1
  return x

def float_to_int(x, levels=256):
  '''normalize float in range [-1, 1] to int in range 0, levels-1
     
     Returns
     -----------
     integers in float data type
  '''
  assert torch.min(x)>=-1 and torch.max(x)<=1, "Input is not in the range -1, 1"
  delta = 2 / levels
  z = torch.clamp(torch.floor((x+1) / delta), min=0, max=levels-1)
  #z = (x/2+0.5)*(levels-1)
  #z = torch.clamp(torch.round(z), min=0, max=levels-1)
  return z

def float_round_int(x, full_range_input=False):
  ''' clamp float to range [-1, 1] first, then convert it to round uint8 data type
      `full_range_input` indicates whether the input is in range [-1, 1], otherwise assumed to be in range [0, 1]
  '''
  if full_range_input:
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2 
  else:
    x = torch.clamp(x, 0, 1)
  x = torch.round(x * 255).int()
  return x

###############################################
# helper functions

# get number of parameters of a model
def get_num_params(model):
  return np.sum(p.numel() for p in model.parameters())


def obj_update(tgt, src):
  '''update the attributes in tgt by the fields of src
     if a attribute doesn't exist in tgt, it will be updated
     by the values in src
  '''
  for k1, v1 in src.__dict__.items():
    if not k1.startswith('_') and not hasattr(tgt, k1):
      setattr(tgt, k1, v1)

# conditional context
# used together with with
def if_context(cond, ctx):
  if cond: return ctx
  return nullcontext()

def init_env(args, log_name=None):
  '''Initialize environment
  '''
  if log_name is not None: logging.getLogger(log_name).info(vars(args))
  vcmrs.log('--------------------------------------------')
  s = pprint.pformat(vars(args))
  vcmrs.log(s)
  vcmrs.log('--------------------------------------------')


  os.makedirs("runs", exist_ok=True)
  os.makedirs("data", exist_ok=True)

  # reproducibility
  if hasattr(args, "trainer"):
    seed = args.trainer.seed
  else:
    seed = args.seed
  torch.manual_seed(seed)
  np.random.seed(seed)


def load_checkpoint(checkpoint_path, model=None, optimizer=None, lr_scheduler=None, alien_weights=False):
  ''' Load the states to the input objects that are not `None`, if possible, from the checkpoint given by `checkpoint_path`.
    Returns:

    `checkpoint`: the checkpoint data itself
  '''

  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  if model is not None:
    assert not isinstance(model, torch.nn.DataParallel), "The model to be loaded is expected to be of class `nn.Module`"

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # vcmrs.debug("Loaded model weights.")

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    vcmrs.debug("Loaded optimizer state.")

  if lr_scheduler is not None and "lr_scheduler" in checkpoint:
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    vcmrs.debug("Loaded learning rate scheduler state.")

  vcmrs.debug("=> loaded checkpoint '{}'".format(checkpoint_path))
  return checkpoint

def count_parameters(model):
  """
  Returns the params count of the given model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_choices(from_list, n_choices):
    return np.random.choice(from_list, size=n_choices, replace=False)

def dig_attr_if_exists(obj, attr_str, default=False):
  warn(f"`dig_attr_if_exists` is deprecated due to misleading name, please use `getattr_by_path` instead.")
  return getattr_by_path(obj,attr_str,default)

def getattr_by_path(obj, attr_str, default=False):
  """Find the attr with the given attribute path, e.g., `'parent.child1.child2'`"""
  attr_path = attr_str.split(".")
  for attr in attr_path:
    obj = getattr(obj, attr, False)
    if not obj:
      return default
  return obj

class RunningAverage:
    """
        Running average utility that keeps track of an arbitary number of values defined by every update call.
    """
    def __init__(self) -> None:
        self._avg_values =  defaultdict(float)
        self._count = 0
             
    def update(self, n=1, **kwargs):
        """
            To define a new value to be tracked, simply add a new key pair value as a parameter of the function.
            Any new key given will be considered as a value that has 0 sum so far.
            The `n` values is the count of the value batch, useful e.g. in case of updating the average losses with the loss sums of a batch of 10.
            Example:
            
            >>> avg.update(n=1, bpp=0.3, mse=0.1, new_key=0.1)
        """
        for k,v in kwargs.items():
            self._avg_values[k] = self._avg_values[k]*self._count + v
            self._avg_values[k] /= self._count + n
        self._count += n
        
    def get_value(self, key):
        return self._avg_values[key]
    
    def get_value_dict(self):
        return dict(self._avg_values) #No more default values
