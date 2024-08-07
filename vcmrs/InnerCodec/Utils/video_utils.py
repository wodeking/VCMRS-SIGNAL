# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import os
import glob
from types import SimpleNamespace

from vcmrs.Utils import data_utils

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

  elif os.path.isfile(fname) and (os.path.splitext(fname)[1].lower() in ['.png', '.jpg', '.jpeg']): 
    video_info.color_space = 'rgb'
    video_info.frame_fnames = [fname] 
    video_info.num_frames = 1
    video_info.resolution = data_utils.get_img_resolution(fname)
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



