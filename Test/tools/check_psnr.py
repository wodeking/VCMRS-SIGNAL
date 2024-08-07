#!/usr/bin/env python

# check the psnr of two images agains a give threshold value

import hashlib
import json
import cv2
import numpy as np
import os
import sys
import glob
import argparse
from vcmrs.Utils import data_utils

parser = argparse.ArgumentParser(description='compare two images against a threshold')
parser.add_argument('--f1', type=str, help='the first file')
parser.add_argument('--f1_bitdepth', type=int, help='bitdepth of f1 if f1 is in YUV format', default=8)
parser.add_argument('--f1_frameskip', type=int, help='skip frame for f1 video', default=0)
parser.add_argument('--f1_frames', type=int, help='number of frames in f1 video, default 0, means all frames', default=0)
parser.add_argument('--f2', type=str, help='the second file')
parser.add_argument('--f2_bitdepth', type=int, help='bitdepth of f2 if f1 is in YUV format', default=8)
parser.add_argument('--threshold', '-t', type=int, default=10, help='threshold in PSNR')

args = parser.parse_args()

def get_psnr(f1, f2):
  img1 = cv2.imread(f1).astype(float)
  img2 = cv2.imread(f2).astype(float)
  if img1.shape != img2.shape:
   print('Error: The resolution does not match')
   print(f1, img1.shape)
   print(f2, img2.shape)
   raise ValueError

  mse = np.mean((img1-img2)**2)
  psnr = 10 * np.log10(255**2 / (mse+1E-50))
  return psnr

def get_md5(fp):
  return hashlib.md5(cv2.imread(fp).tobytes()).hexdigest()

def check_md5(f1,f2):
  img1 = get_md5(f1)
  if isinstance(f2, dict):
    img2 = f2[os.path.basename(f1)]
  else:
    img2 = get_md5(f2)
  return img1 == img2

def check_md5_file(f1, f2):
  with open(f1, 'rb') as f:
    c1 = hashlib.md5(f.read()).hexdigest()
  with open(f2, 'rb') as f:
    c2 = hashlib.md5(f.read()).hexdigest()
  return c1==c2

def check_yuv_files(f1, f2, f1_bitdepth, f2_bitdepth, f1_frameskip=0, f1_frames=0):
  # get width and height from yuv file name
  W, H = data_utils.get_seq_info(f1)
  if f1_frames == 0:
    num_frames1 = data_utils.get_num_frames_from_yuv(f1, W, H, chroma_format='420', bit_depth=f1_bitdepth)
  else:
    num_frames1 = f1_frames
  num_frames2 = data_utils.get_num_frames_from_yuv(f2, W, H, chroma_format='420', bit_depth=f2_bitdepth)
  assert num_frames1 == num_frames2, f"Number of frames does not match, {num_frames1} vs {num_frames2}"
  worst_psnr = 1E20
  for frame_idx in range(num_frames1):
    frame1 = data_utils.get_frame_from_raw_video(f1, frame_idx+f1_frameskip, \
      W=W, 
      H=H, 
      chroma_format='420', 
      bit_depth=f1_bitdepth, 
      is_dtype_float=False)
    frame2 = data_utils.get_frame_from_raw_video(f2, frame_idx, \
      W=W, 
      H=H, 
      chroma_format='420', 
      bit_depth=f2_bitdepth, 
      is_dtype_float=False)
    mse = np.mean((frame1-frame2)**2)
    psnr = 10 * np.log10(255**2 / (mse+1E-50))
    print(f"frame {frame_idx}: {psnr}")
    worst_psnr = min(worst_psnr, psnr)
  return worst_psnr



def check_files(f1, f2, f1_bitdepth=0, f2_bitdepth=0):
  if args.threshold == 0: # Bit exact
    if os.path.splitext(f1)[1].lower() == '.yuv':
      if check_md5_file(f1, f2):
        print('Bit-exactness verified.')
      else:
        print(f"Error! {f1} and {f2} does not match")
        sys.exit(1)
    else:
      # png file
      if check_md5(f1, f2):
        print("Bit-exactness verified.")
      else:
        print(f'Error! {f1} and {"the decoded file" if isinstance(f2, dict) else f2} are not bit-exact!')
        sys.exit(1)
  else:
    if os.path.splitext(f1)[1].lower()=='.yuv':
      # compare yuv file
      psnr = check_yuv_files(f1, f2, f1_bitdepth, f2_bitdepth, args.f1_frameskip, args.f1_frames)
    else:
      psnr = get_psnr(f1, f2)
    print('Matching images, PSNR:', psnr)
    if psnr < args.threshold:
       print(f'Error! PSNR={psnr} is smaller than threshold {args.threshold}')
       sys.exit(1)


data = None

if os.path.isdir(args.f1):
  f1_files = sorted(glob.glob(os.path.join(args.f1, '*')))

  if data is not None:
    f2_files = [data] * len(f1_files)
  else:
    f2_files = sorted(glob.glob(os.path.join(args.f2, '*')))

  if len(f1_files) != len(f2_files):
    print(f'Error! The two folders do not contain the same number of files. {len(f1_files)} vs {len(f2_files)}')
  for f1, f2 in zip(f1_files, f2_files):
    check_files(f1, f2)
  
else:
  #f2 = data if data else args.f2
  check_files(args.f1, args.f2, args.f1_bitdepth, args.f2_bitdepth)


