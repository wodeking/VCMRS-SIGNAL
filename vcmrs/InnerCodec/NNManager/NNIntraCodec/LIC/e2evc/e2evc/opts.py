# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any

import yaml

def get_arguments():
  '''Get input arguments

  Returns
  -----------------
  args
    a dictionary of arguments
  '''

  
  parser = get_parser()
  args = parser.parse_args()

  return args

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_name', type=str, default=None, help='The name of the experiment')
  parser.add_argument('--tasks', default=["seg", "det"], nargs="*", type=str)
  parser.add_argument('--data_dir', type=str, default='data', 
                      help='Location for the dataset')
  parser.add_argument('--dataset', type=str, default='cifar', 
                      help='Can be either cifar|imagenet64|ucf101|image_folder|csv_file')
  parser.add_argument('--num_workers', type=int, default=8, help='Number of threads when loaded the data')
  parser.add_argument('--subset_lengths', default=[-1, -1], nargs=2, type=int, help="Number of items for the subsets for train and validation respectively. -1 means all.")
  parser.add_argument('--color_space', type=str, default='rgb',choices=['rgb', 'yuv420'], help='convert input image to defined color space')
  parser.add_argument('-b', '--batch_size', type=int, default=12,
                      help='Batch size during training per GPU')
  return parser

def read_yaml(configs_path=None) -> None:
  """Read the nested dicts from a yaml file and make the properties accessible without the bracket syntax.
  """
  # def __init__(self, configs_path=None) -> None:
  with open(configs_path, 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
  return json.loads(json.dumps(configs), object_hook=lambda d: SimpleNamespace(**d))

if __name__ == "__main__":
  a = read_yaml(configs_path="/raid/nam/E2E_repos/vcm_e2e_image/e2evc/configs/GANs_finetune.yaml")
  b=a.unknown
  b=a.model.content_loss
  b = a.model
  b= a.optimizer
  b = a.model.unknown2
  a.hola = 1
  a.model.a = 2
  a.model.update_attrib("aca",00)
  a.update_attrib("model",00)
  a.update_attrib("model",1)
  # a.adsd.update_attrib("asd", 0) # Should crash
  # a.model = 1 # Should crash
  b=0

