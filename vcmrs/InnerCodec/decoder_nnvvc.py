# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import os
import argparse
import pprint
import asyncio
from functools import partial

from .NNManager.Main import nnmanager as nnmanager
from .Utils import nnmanager_utils
from .Utils import dec_utils

import vcmrs
from vcmrs.Utils import utils
from vcmrs.Utils import io_utils
from vcmrs.Utils.codec_context import CodecContext

def get_arguments():
  '''Get input arguments

  Args:
    parser: parser object
  '''
  parser = argparse.ArgumentParser(description='Encode video or images for machine consumption')

  # data I/O
  parser.add_argument('--input_prefix', type=str, default='', 
                      help='Prefix, e.g., directory of the input data. Default: emtpy.')

  parser.add_argument('input_files', type=str, nargs='*', default='', 
    help='Input bitstream file name, or a directory that contains bitstream files')

  parser.add_argument('--output_dir', type=str, default='./output', 
                      help='Directory of the output data. Default: output.')

  parser.add_argument('--output_frame_format', type=str, default='frame_{frame_idx:06d}.png', 
                      help='The output frame file name format for video filese. data. Default: frame_{frame_idx:06d}.png, which will generage frames with file names like frame_000000.png, frame_000001.png, .... One can also use variable frame_idx1 for frame index starting from 1, for example {frame_idx1:06d}.png. ')

  # internal configurations
  parser.add_argument('--working_dir', type=str, default=None,
                      help='working directory to store temporary files')
  parser.add_argument('--cvc_dir', type=str, default=os.path.join(\
                        os.path.dirname(os.path.dirname(__file__)), 'Tools', 'vtm_lcvc'),
                      help='directory of the CVC encoder directory')
  parser.add_argument('--port', type=str, default='*',
                      help='Port number for internal communication. Default: 6734')
  parser.add_argument('--debug', action='store_true', 
                      help='In debug mode, more logs are printed, intermediate files are not removed')
  parser.add_argument('--logfile', default='', type=str, help='Path of the file where the logs are saved. By default the logs are not saved.')

  args = parser.parse_args()

  return args



def initialize(ctx):
  # make deterministic
  utils.make_deterministic(True)

  # start NNManager server
  nnmanager_utils.nnmanager_start(ctx)

  ctx.inner_ctx.nnmanager_lock = asyncio.Lock() # 1 request to NN Manager

def exit_handler(ctx):
  # stop the nnmanager server
  nnmanager_utils.nnmanager_stop(ctx)
  
def process(input_files, ctx):
  initialize(ctx)
  for item in input_files:
    dec_utils.process_file(item, ctx)

def main(args):
  r"""
  Main function.
  """
  args = get_arguments()
  initialize(args)

  # get input files
  input_files = io_utils.get_input_files(args)

  ctx = CodecContext(args)
  process(input_files, ctx)
  exit_handler(ctx)

if __name__ == '__main__':
  args = get_arguments()
  s = pprint.pformat(args.__dict__)
  vcmrs.log(s)

  main(args)

