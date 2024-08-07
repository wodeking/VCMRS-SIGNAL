# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import time
import pprint
import asyncio

from .NNManager.Main import nnmanager as nnmanager
from .Utils import nnmanager_utils
from .Utils import enc_utils

import vcmrs
from vcmrs.Utils import encoder_opts
from vcmrs.Utils import io_utils
from vcmrs.Utils.codec_context import CodecContext



def initialize(ctx):
  r"""
  Initialize the encoding environment.

  Args:
    args: input arguments
  """
  # start NNManager server
  nnmanager_utils.nnmanager_start(ctx)


def exit_handler(ctx):
  # stop the nnmanager server
  nnmanager_utils.nnmanager_stop(ctx)

def process(items, ctx):
  initialize(ctx)
  asyncio.run(enc_utils.process_input(items, ctx))
  
def main(args):
  r"""
  Main function.
  """
  args = encoder_opts.get_encoder_arguments()
  vcmrs.setup_logger("main_encoder", args.logfile, args.debug)
  vcmrs.log("Start encoding...")
  vcmrs.log('Input arguments: ')
  s = pprint.pformat(args.__dict__)
  vcmrs.log(s)

  # start_time 
  start_time = time.time()
  ctx = CodecContext(args)

  input_files = io_utils.get_input_files(args)
  process(input_files, ctx)

  elapse = time.time()-start_time
  vcmrs.log(f"Encoding completed in {elapse} seconds")
  exit_handler(ctx)


if __name__ == '__main__':
  main()

