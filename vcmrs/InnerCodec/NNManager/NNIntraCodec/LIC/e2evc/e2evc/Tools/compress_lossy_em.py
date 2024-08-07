# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

# compress an image using a lossy model based on entropy model

import time
import os
import numpy as np
import torch
from pathlib import Path
from e2evc import Model
from e2evc.ProbModel import PMSModel
from e2evc.Utils import utils as e2eutils
from e2evc.Tools.cmpr_lossy_pms_utils import compress as compress_pms, decompress as decompress_pms
from e2evc.Datasets.transform_utils import batch_images_fn
from torchvision.utils import save_image
import torch
import vcmrs
from e2evc.opts import read_yaml, get_parser


def get_arguments():
  parser = get_parser()
  parser.add_argument('--epoch',type=int,
                      help='Epoch model. If not specified, it will be all.')
  parser.add_argument('--output_path', type=str, default=os.getcwd(), help='output directory')
  parser.add_argument('--file_list',type=str,
                      help='File list')
  parser.add_argument('--pretrained', type=str, help='Pretrained model file name.')
  parser.add_argument("--save_bitstream", action="store_true")
  parser.add_argument('--model_config', type=str, help='YAML file that contains the model configs, overrides the other command line arguments regarding the model architecture if provided')
  
  args = parser.parse_args()
  return args

def compress_one_image(model, i_imgfp, o_bsfp):
  model.eval()
  i_image = e2eutils.load_img_tensor(i_imgfp)
  original_size = i_image.shape[1:]
  i_image = batch_images_fn(images=i_image, size_divisible=model.size_divisible)
  i_image = i_image.to(next(model.parameters()).device) # Send the image to the model's device
  os.makedirs(os.path.dirname(o_bsfp), exist_ok=True)

  with torch.no_grad():
    bstr_bpp = compress_pms(model, i_image, o_bsfp)
    padded_pixels = i_image.shape[-2:].numel()
    origin_pixels = original_size[-2:].numel()
    bstr_bpp = (bstr_bpp * padded_pixels)/origin_pixels
  return {
    "original_size": list(original_size),
    "bitstream_bpp": bstr_bpp
  }

def decompress_one_image(model, i_bsfp, o_imgfp, original_size):
  model.eval()
  os.makedirs(os.path.dirname(o_imgfp), exist_ok=True)
  # torch.cuda.empty_cache()

  with torch.no_grad():
    # log("Start decompressing...")
    
    x_hat = decompress_pms(model, i_bsfp, original_size)

    x_tilde = x_hat.squeeze().type(torch.uint8) # C H W
    x_tilde = x_tilde[:original_size[0],:original_size[1],:original_size[2]]
    x_tilde = x_tilde.permute(1,2,0) # H W C
    e2eutils.save_img(x_tilde, o_imgfp)


def build_intra_codec(args):
  rets = {}
  input_check_fun = lambda x: x.min() >= 0 and x.max() <= 1
  model = Model.ModelFactory.create_model(args.model_configs.codec)
  model_fname=args.model_fname
  if os.path.exists(model_fname):	
    checkpoint = e2eutils.load_checkpoint(model_fname, model=model)	
  else:	
    vcmrs.error(f"{model_fname} is not found.")	
    raise FileNotFoundError(model_fname)
  vcmrs.debug(f'Model {args.model_configs.codec.model} loaded')

  rets.update({
    "codec": model,
    "input_check_fun": input_check_fun,
  })
  
  return rets
