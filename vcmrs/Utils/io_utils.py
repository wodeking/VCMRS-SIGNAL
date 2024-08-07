# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# utilities

import sys 
import tempfile

import os
import glob
import copy
import shlex
import shutil
from types import SimpleNamespace
import configparser
import yaml
import json
import tempfile
import time

import numpy as np

from . import encoder_opts
from . import data_utils
import vcmrs

###############################################
# input and output
class FileItem():
  def __init__(self, fname, args):
    # member variables
    self.fname = fname        # input file
    self.args = copy.deepcopy(args)
    # self.args = args          # input arguments
    self.working_dir = None   # working directory
    #self.bitstream_dir = None # output bitstream filename
    #self.recon_dir = None     # output directory for reconstructed image or video

    # parameters to be included in the bitstream
    self.parameters = {}
    # control parameters to inner codec
    self.inner_control_signals = {}


    self._bname = os.path.basename(fname) # input files' basename
    self._is_yuv_video = fname[-3:].lower() == 'yuv' 
    self._is_dir_video = os.path.isdir(fname)   # input file is a directory (video)
    self._is_video = self._is_yuv_video or self._is_dir_video


    ######################################################
    # for inner codec
    self.inner_in_fname = None  # input file name for inner codec
    self.inner_out_fname = None # output file name for inner codec

    # set working dir
    if not args.working_dir:
      self.working_dir = tempfile.mkdtemp()
 
    wd = os.path.join(args.working_dir, self._bname)
    os.makedirs(wd, exist_ok=True)
    self.working_dir = wd

    if hasattr(args, 'output_bitstream_fname'): # encoder 
      bname = os.path.splitext(self._bname)[0]
      self.bitstream_fname = os.path.join(args.output_dir, args.output_bitstream_fname.format(bname=bname))

      #if not self.bitstream_fname:
      #  self.bitstream_fname = os.path.join(args.output_dir, 'bitstream', os.path.splitext(self._bname)[0]+'.bin')
      #else:
      #  self.bitstream_fname = os.path.join(args.output_dir, args.output_bitstream_fname)

      os.makedirs(os.path.dirname(self.bitstream_fname), exist_ok=True)

  def __str__(self):
    """For easier debugging"""
    for attribute, value in vars(self).items():
      if value:
        print(f"{attribute}: {value}")
  
  # add parameter to the bitstream 
  def add_parameter(self, component_name, param_data=None):
    # check component_name
    if component_name in self.parameters.keys():
      assert False, f"Parameters for {component_name} has already been added"

    assert type(param_data)==bytearray, "param_data must be a type of bytearray"
     
    self.parameters[component_name] = param_data

  # get parameters for a component
  def get_parameter(self, component_name):
    if component_name in self.parameters.keys():
      return self.parameters[component_name]
    return None

  def add_inner_control_signal(self, component_name, signal=None):
    # check component_name
    if component_name in self.inner_control_signals.keys():
      assert False, f"Parameters for {component_name} has already been added"

    self.inner_control_signals[component_name] = signal 

  # get parameters for a component
  def get_inner_control_signal(self, component_name):
    if component_name in self.inner_control_signals.keys():
      return self.inner_control_signals[component_name]
    return None

  ##################################################################
  # get ouput file name or directory name for pre-inner codec components
  def get_stage_output_fname(self, stage):
    sd = os.path.join(self.working_dir, stage)
    os.makedirs(sd, exist_ok=True)
    if self._is_video:
      out_fname = os.path.join(sd, self._bname)
    elif self._is_dir_video:
      out_fname = os.path.join(sd, self._bname)
      os.makedirs(out_fname, exist_ok=True)
    else:
      out_fname = os.path.join(sd, os.path.splitext(self._bname)[0]+'.png')
    return out_fname

  def get_stage_output_fname_decoding(self, stage, img=False, format='YUV'):
    sd = os.path.join(self.working_dir, stage)
    os.makedirs(sd, exist_ok=True)
    if img: 
      return os.path.join(sd, os.path.splitext(self._bname)[0]+'.png')

    if format=='YUV':
      out_fname = os.path.join(sd, os.path.splitext(self._bname)[0]+'.yuv')
    else:
      out_fname = os.path.join(sd, self._bname)
      os.makedirs(out_fname, exist_ok=True)
    return out_fname


# encoder input file handling
def get_input_files(args):
  input_files = []

  for fname in args.input_files:
    fname = args.input_prefix + fname
    if os.path.isdir(fname):
      if args.directory_as_video:
        input_files.append(FileItem(fname, args))
      else:
        input_files += [FileItem(x, args) for x in sorted(glob.glob(os.path.join(fname, '*')))]

    elif is_ini_file(fname):
      input_files += parse_ini_file(fname, args)

    elif os.path.isfile(fname):
      # handle input image/video file
      input_files.append(FileItem(fname, args))
    else: 
      vcmrs.log(f"Cannot open input file: {fname}")
      sys.exit(1)

  return input_files

# input file handling
def decoder_get_input_files(args):
  input_files = []
  for fname in args.input_files:
    if  os.path.isdir(fname):
      input_files += sorted(glob.glob(os.path.join(fname, '*')))
    else:
      input_files.append(fname)
  items = [FileItem(x, args) for x in input_files]
  return items 


def is_ini_file(fname):
  return os.path.splitext(fname)[1].lower() in ['.ini']

def parse_ini_file(fname, system_args):

  cfg = configparser.ConfigParser()
  cfg.optionxform=str # case sensitive
  cfg.read(fname)

  def parse_sec(sec):
    args = []
    for key in sec:
      args.append('--'+key)
      if sec[key]: args.append(sec[key])
    return args

  # default section contains configurations for all other sections
  default_args = []
  if 'default' in cfg.sections():
    default_args = parse_sec(cfg['default'])
  
  input_files = []
  for sec in cfg.sections():
    if sec == 'default': continue
    
    args = []
    args += default_args
    args += parse_sec(cfg[sec])
    args += shlex.split(sec)

    ini_args = encoder_opts.get_encoder_arguments(args, system_args)

    input_files += get_input_files(ini_args)

  return input_files 


def gen_output_recon(input_fname, item):

  bname = os.path.splitext(item._bname)[0]
  output_recon_fname = os.path.join(item.args.output_dir, item.args.output_recon_fname.format(bname=bname))

  #if item._is_dir_video:
  if os.path.isdir(input_fname):
    o_dir = output_recon_fname
 
    fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
    os.makedirs(o_dir, exist_ok=True)
    for frame_idx, fname in enumerate(fnames):
      frame_fname = item.args.output_frame_format.format( \
       frame_idx=frame_idx,
       frame_idx1=frame_idx+1)
      o_fname = os.path.join(o_dir, frame_fname)
      shutil.copy(fname, o_fname, follow_symlinks=True)
  else:

    #if item._is_yuv_video:
    if os.path.splitext(input_fname)[1].lower()=='.yuv':
      o_fname = output_recon_fname + '.yuv'
    elif os.path.splitext(input_fname)[1].lower()=='.png':
      # image
      o_fname = output_recon_fname + '.png'
    else:
      assert False, f"unsupported stage file name {input_fname}"

    os.makedirs(os.path.dirname(o_fname), exist_ok=True)
    shutil.copy(input_fname, o_fname, follow_symlinks=True)

# get number of frames from a yuv video file or a directory of frames
# return number of frames and frame file names
# if the YUV file is given, the frame file names is the YUV file name
def get_video_info(fname, args):
  video_info = SimpleNamespace()
  video_info.frame_rate = args.FrameRate
  video_info.bit_depth = args.InputBitDepth
  video_info.chroma_format = args.InputChromaFormat
  video_info.yuv_full_range = args.InputYUVFullRange
  
  if os.path.isdir(fname):
    video_info.color_space = 'rgb'
    video_info.frame_fnames = sorted(glob.glob(os.path.join(fname, '*.png')))
    video_info.num_frames = len(video_info.frame_fnames)
    video_info.resolution = data_utils.get_img_resolution(video_info.frame_fnames[0])
  elif os.path.isfile(fname) and (os.path.splitext(fname)[1].lower() == '.yuv'): 
    video_info.color_space = 'yuv'
    video_info.frame_fnames = fname
    video_info.num_frames = data_utils.get_num_frames_from_yuv(
      fname,
      args.SourceWidth,
      args.SourceHeight,
      args.InputChromaFormat,
      args.InputBitDepth)
    video_info.resolution = (args.SourceHeight, args.SourceWidth, 3) # (H,W,C)
  else:
    assert False, f'Input is not a video YUV file or a directory contraining frames {fname}'
  return video_info

# check if video info is a YUV video file
def is_yuv_video(video_info):
  return (type(video_info.frame_fnames) == str) and \
    (os.path.splitext(video_info.frame_fnames)[1].lower() == '.yuv')

# get pix_fmt for ffmpeg from video info
def get_ffmpeg_pix_fmt(video_info):
  assert video_info.color_space == 'yuv', 'Only video in YUV color space is suported'
  assert video_info.bit_depth in [8, 10], 'Only bit depth 8 or 10 is supported'

  if video_info.bit_depth == 10:
    if video_info.chroma_format == '420': 
      pix_fmt = 'yuv420p10le'
    else:
      pix_fmt = 'yuv444p10le'
  else:
    if video_info.chroma_format == '420': 
      pix_fmt = 'yuv420p'
    else:
      pix_fmt = 'yuv444p'
  return pix_fmt


##############################################
def read_yaml(configs_path=None) -> None:
  """Read the nested dicts from a yaml file and make the properties accessible without the bracket syntax.
  """
  # def __init__(self, configs_path=None) -> None:
  with open(configs_path, 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
  return json.loads(json.dumps(configs), object_hook=lambda d: SimpleNamespace(**d))

##############################################
def create_temp_folder_suffix(temp_folder):
  if not os.path.exists(temp_folder):
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder
  else:
    suffix = 1
    while True:
      temp_folder_path = f"{temp_folder}_{suffix}"
      if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path, exist_ok=True)
        return temp_folder_path
      suffix += 1

##############################################
def enforce_symlink(src, dst, target_is_directory = False):
  """wrapper for os.symlink which assures creation over network-mapped drives
     usage:  enforce_symlink(src="some_file.txt", dst="resultant_symlink")
  """
  
  #if relative_symlink:
  #  src = os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst)))
  
  for i in range(10,0, -1):
    rm_file_dir(dst)
   
    if i>1:
      try:
        os.symlink(src, dst)
        return
      except:
        time.sleep(2)
    else: # last try - maybe OS is not supporting symlinks?
      #os.symlink(src, dst)
      shutil.copyfile(src, dst)
      return

def rm_file_dir(dst):
  """Remove a file, directory or a symbolic link
  """
  if os.path.islink(dst):
    os.unlink(dst)
  elif os.path.isdir(dst):
    #os.rmdir(dst)
    shutil.rmtree(dst)
  elif os.path.isfile(dst):
    os.remove(dst)
  else:
    pass

 
