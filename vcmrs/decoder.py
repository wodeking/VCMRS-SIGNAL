# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import os
import argparse
import tempfile
import atexit
import time
import cv2

from vcmrs.Utils import component_utils
from vcmrs.Utils import io_utils
from vcmrs.Utils import utils
from vcmrs.Utils.codec_context import CodecContext
from vcmrs.InnerCodec import decoder_nnvvc
import vcmrs

def get_arguments():
  '''Get input arguments

  Args:
    parser: parser object
  '''
  parser = argparse.ArgumentParser(description='Decode video or images for machine consumption')

  # data I/O
  parser.add_argument('--input_prefix', type=str, default='', 
                      help='Prefix, e.g., directory of the input data. Default: emtpy.')

  parser.add_argument('input_files', type=str, nargs='*', default='', 
    help='Input bitstream file name, or a directory that contains bitstream files')

  parser.add_argument('--output_dir', type=str, default='./output', 
                      help='Directory of the output data. Default: output.')
  parser.add_argument('--output_recon_fname', type=str, default='{bname}', 
                      help='file name for the output reconstructed data. Default empty string which means the output recon file name is the same as the input image or video bitstream file name, in {output_dir} directory.')

  parser.add_argument('--output_video_format', type=str, default='YUV', 
                      help='output video format, can be YUV or PNG. Default is YUV.')
  parser.add_argument('--output_frame_format', type=str, default='frame_{frame_idx:06d}.png', 
                      help='The output frame file name format for video filese in PNG. Default: frame_{frame_idx:06d}.png, which will generage frames with file names like frame_000000.png, frame_000001.png, .... One can also use variable frame_idx1 for frame index starting from 1, for example {frame_idx1:06d}.png. ')

  parser.add_argument('--single_frame_image', action='store_true', 
                      help='Treat single frame output as a image. For image output, the decoder outputs file in png format with name <bistream_basename.png>')

  # codec tools configuration
  # Note: for testing purpose only. Will be removed once the tools are determined. 

  parser.add_argument('--TemporalResample', type=str, default="resample",
                      help='Method for TemporalResample component')

  parser.add_argument('--SpatialResample', type=str, default="Bypass",
                      help='Method for SpatialResample component')

  parser.add_argument('--ROI', type=str, default="roi_generation",
                      help='Method for ROI component')

  parser.add_argument('--InnerCodec', type=str, default="NNVVC",
                      help='Method for InnerCodec component, can be NNVVC or VTM')

  parser.add_argument('--IntraCodec', type=str, default='LIC', 
                      help='Method for end-to-end intra codec')
  
  parser.add_argument('--BitDepthTruncation', type=str, default="truncation",
                      help='Method for PostFilter component')
  
  parser.add_argument('--PostFilter', type=str, default="Bypass",
                      help='Method for PostFilter component')

  parser.add_argument('--VCMBitStructOn', type=int, default=1, 
                      help='1 for turning on the new VCM bitstream structure design and 0 for turning off')

  # internal configurations
  parser.add_argument('--working_dir', type=str, default=None,
                      help='working directory to store temporary files')
  parser.add_argument('--cvc_dir', type=str, default=os.path.join(\
                        os.path.dirname(__file__), 'InnerCodec', 'VTM'),
                      help='directory of the CVC encoder directory')
  parser.add_argument('--port', type=str, default='*',
                      help='Port number for internal communication. Default: 6734')
  parser.add_argument('--debug', action='store_true', 
                      help='In debug mode, more logs are printed, intermediate files are not removed')
  parser.add_argument('--debug_source_checksum', action='store_true', 
                      help='Print md5 checksums of source files')
  parser.add_argument('--logfile', default='', type=str, help='Path of the file where the logs are saved. By default the logs are not saved.')
  parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                      help='Path to ffmpeg executable')

  args = parser.parse_args()

  return args



def initialize(args):
  if not args.working_dir:
    args.working_dir = tempfile.mkdtemp(dir=args.working_dir)

  # make context  
  ctx = CodecContext(args)
  global _ctx
  _ctx = ctx

  # parse input files
  ctx.input_files = io_utils.decoder_get_input_files(args)

  # register exit handler to terminate nnmanager
  atexit.register(exit_handler)

  # make torch working in deterministic mode
  utils.fix_random_seed()
  utils.make_deterministic()
  
  return ctx

def exit_handler():
  global _ctx

  # inner codec clean
  decoder_nnvvc.exit_handler(_ctx)

  # clean working directory
  if not _ctx.input_args.debug:
    for item in _ctx.input_files:
      io_utils.rm_file_dir(item.working_dir)
 
def decode(args):
  # initialize
  ctx = initialize(args)

  for item in ctx.input_files:
    item.inner_in_fname = item.fname
    #set by the inner codec
    #item.inner_out_fname = item.get_stage_output_fname(f"inner")

  # inner codec
  s_time = time.time()
  decoder_nnvvc.process(ctx.input_files, ctx)
  vcmrs.log(f"[{os.path.basename(item.args.working_dir)}] Inner decoding done. Time = {(time.time() - s_time):.6}(s)")

  # post-inner components 
  post_components = ['ROI', 'SpatialResample', 'TemporalResample', 'PostFilter', 'BitDepthTruncation']
  for item in ctx.input_files:
    vcmrs.log(f'Post-inner processing file: {item.fname}')
    
    item.FrameRateRelativeVsInnerCodec = 1.0

    in_fname = item.inner_out_fname
    for c_name in post_components:
      component_method = getattr(ctx.input_args, c_name)
      component = component_utils.load_component(c_name, "decoder", component_method, ctx)
      #out_fname = item.get_stage_output_fname_decoding(f"post_{c_name}")
      out_fname = os.path.join(os.path.dirname(os.path.dirname(in_fname)), f"post_{c_name}", os.path.basename(in_fname))
      component.process(os.path.abspath(in_fname), os.path.abspath(out_fname), item, ctx)
      in_fname = out_fname
      component = None # free memory after each iteration
   
    # generate ouptut recon
    io_utils.gen_output_recon(in_fname, item)


def main():
  r"""
  Main function.
  """
  args = get_arguments()
  vcmrs.setup_logger("main_decoder", args.logfile, args.debug)
  if args.debug_source_checksum:
    checksums = utils.get_project_checksums()
    vcmrs.log("Decoder checksums: \n" + "\n".join(checksums))

  vcmrs.log("Start decoding...")

  cv2.setNumThreads(0) # disables spawning of many threads for load/write operations

  start_time = time.time()
  decode(args)

  elapse = time.time()-start_time

  vcmrs.log(f'Decoding completed in {elapse} seconds')


if __name__ == '__main__':
  main()

