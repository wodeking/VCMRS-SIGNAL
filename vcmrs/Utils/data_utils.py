# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# utilities for media data, for example, data io, format conversion

#from os import path
#this_path = path.dirname(path.abspath(__file__))

import os
import re
import cv2
import numpy as np
import math
import tempfile
from . import utils
from . import io_utils


###########################################
# Media file type support

# cehck if a file is an image
def is_file_image(fname):
  '''check if a file is a image file by file extention
  
  Params
  ------
    fname: input file name

  Return
  ------
    True if the input file has extention .png or .jpg
  '''
  return os.path.splitext(fname)[1].lower() in ['.png', '.jpg']

# check if a file is a video
def is_file_video(fname):
  '''check if a file is a video sequence file by file extention

  Params
  ------
    fname: input file name

  Return
  ------
    True if the input file has extention .yuv
  '''
  return os.path.splitext(fname)[1].lower() in ['.yuv']
  
# get image resolution
def get_img_resolution(fname):
  '''get resolution of a image file
  Params
  ------
    fname: input file name. Must be an image file that can be loaded by cv2

  Return
  ------
    H,W,C 
  '''
  img = cv2.imread(fname)
  H,W,C = img.shape
  return H, W, C

###########################################
def get_image_file_list(path):
  if path is None: return None
  
  file_list = os.listdir(path)

  # filter the list to only include .png files
  png_list = [file for file in file_list if is_file_image(file)]
  
  # sort the list in ascending order
  png_list.sort()
  return png_list

###########################################
# YUV input data support

# get size in bytes of each frame in a YUV file
def get_frame_size_from_yuv(
    SourceWidth,
    SourceHeight,
    InputChromaFormat='420', 
    InputBitDepth=8):
  '''calculate the size of each frames in a YUV sequence
  
  Params
  ------
    width: width
    height: height
    chroma_format: chroma format, either '420' or '444'
    bit_depth: bit depth, either 8 or 10
  '''
  bytes_per_pixel = 3 if InputChromaFormat=='444' else 1.5
  if InputBitDepth == 10: bytes_per_pixel *= 2
  frame_size = int(SourceWidth * SourceHeight * bytes_per_pixel)
  return frame_size

# get number of frames from a YUV video file
def get_num_frames_from_yuv(fname, width, height, chroma_format='420', bit_depth=8):
  '''calculate the number of frames contained in a YUV sequence
  
  Params
  ------
    width: width
    height: height
    chroma_format: chroma format, either '420' or '444'
    bit_depth: bit depth, either 8 or 10
  '''
  file_size = os.path.getsize(fname)
  if chroma_format == '420':
    assert (width % 2 == 0) and (height % 2 == 0),  \
      print('The width and height must be even number')
  assert (bit_depth in [8, 10]), print('bit_depth must be 8 or 10')
  assert (chroma_format in ['420', '444']), \
    print('chroma_format must be 420 or 444')
  frame_size = get_frame_size_from_yuv(
    width,
    height,
    chroma_format, 
    bit_depth)

  assert file_size % frame_size == 0, \
    print('file size does not match the source format')

  num_frames = int(file_size / frame_size)
  return num_frames

# split yuv video file into frames
def split_yuv_video(fname, output_dir, width, height, chroma_format='420', bit_depth=10):
  '''Split a YUV video sequence into frames, in YUV file format with the same data format

  After the conversion, frames are saved with file names, scuh as frame_[width]x[height]_000001.yuv. The index starts from 0. 
  Params
  ------
    fname: input YUV file name
    output_dir: output directory
    width: the width of the frame
    height: the height of the frame
    chroma_format: either '420' or '444'
    bit_depth: either 8 or 10
  '''
  frame_size = get_frame_size_from_yuv( \
    width, 
    height, 
    InputChromaFormat = chroma_format,
    InputBitDepth = bit_depth)

  frame_idx = 0
  with open(fname, 'rb') as f:
    while True: 
      buf = f.read(frame_size)
      if not buf: break
      assert len(buf) == frame_size, print(f'Size of file {fname} does not match the data format')
      with open(os.path.join(output_dir, f'frame_{width}x{height}_{frame_idx:06d}.yuv'), 'wb') as of:
        of.write(buf)
      frame_idx += 1
    
def get_seq_info(fname):
  ''' get weight and height information from the file name of a sequence. The file name may be
        xxxxxx_[width]x[height]_xxxxxx.xxx

      Params
      ------
        fname: input file name
      Return
      ------
        W,H if width and height is found in the input file name. Otherwise, return 0,0
  '''
  bname = os.path.basename(fname)
  m = re.search('(\d\d+)x(\d\d+)', bname)
  W = H = 0 
  if m:
    W = int(m.group(1))
    H = int(m.group(2))
  return W, H

# get frame data in HW3 format
def get_frame_from_raw_video(fname, frame_idx, \
    W=None, 
    H=None, 
    chroma_format='420', 
    bit_depth=8, 
    is_dtype_float=False,
    return_raw_data=False):
  ''' get a frame in 10bit YUV 444 format from a sequence file. 
      If W and H is not given, it's inferred from the sequeence file name

      Params
      ----------
        fname: file name of the sequence. It may contains <width>x<height> in the name
        frame_idx: the frame to be extracted, starting from 0
        W, H: width and height. If not given, it's inferred from the file name
        chroma_format: '420', '444', default '420' #'rgb', 'rgbp', 'yuv420p', 'yuv420p_10b'
        bit_depth: 8, 10, default 8
        is_dtype_flaot: if true, return float tensor with values in range [-1, 1]
        return_raw_data: if true, retype byte array

      Return
      ------
        Image in format HW3, and in range [-1, 1] if is_dtype_float is set. Otherwise, return data in uint16
  '''
  if W is None or H is None: 
    W,H = get_seq_info(fname)

  dtype = 'uint8'
  scale = 255
  if chroma_format == '444': 
    frame_length = W*H*3
  else: 
    frame_length = W*H*3//2
  if bit_depth == 10:
    frame_length *= 2 
    dtype = 'uint16'
    scale = 1024

  start_pos = frame_idx*frame_length
  with open(fname, 'rb') as f:
    f.seek(start_pos)
    data = f.read(frame_length)

  if return_raw_data: return data

  frame_data = np.frombuffer(data, dtype=dtype)

  if chroma_format=='444':
    # rgb planar
    frame=frame_data.reshape(3,H,W)
    frame = frame.transpose(1, 2, 0)
  else: 
    # yuv420 planar
    y = frame_data[:H*W].reshape(H, W)
    uv_length = H*W//4
    u = frame_data[H*W:H*W+uv_length].reshape(H//2, W//2)
    v = frame_data[H*W+uv_length:].reshape(H//2, W//2)
    # upsample u an v by kronecker product
    kernel = np.array([[1,1], [1,1]], dtype=dtype)

    u = np.kron(u, kernel)
    v = np.kron(v, kernel)
    img_yuv = np.stack([y, u, v])  # 3HW
    frame = img_yuv.transpose(1,2,0) # HW3

  if is_dtype_float: 
    frame = frame.astype(float) / scale * 2 - 1
  elif bit_depth==8:
    frame = frame.astype('uint16')
    frame *= 4

  return frame



###############################################
# data format conversion

def cvt_yuv444p_to_yuv420p10b(in_yuv_data):
  ''' Convert image from yuv444p to yuv420p 10bit format
  
  Params
  ------
    yuv_data: ndarray, in yuv444p format, 10 bit, HW3

  Return
  ------
    Y, U, V: ndarray, with dtype uint16. Y component has shape HW3. U and V 
    components have shape (H//2, W//2, 3)
  '''
  yuv_data = in_yuv_data.transpose(2,0,1) #3HW
  yuv_data = yuv_data.astype('uint16')

  if in_yuv_data.dtype == 'uint8':
    yuv_data *= 4

  yy,uu,vv = yuv_data
  H,W = yy.shape
  uu = uu.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3)).astype('uint16')
  vv = vv.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3)).astype('uint16')
  return yy, uu, vv

def cvt_yuv444p10b_to_bgr(in_yuv_data):
  return cvt_yuv444p_to_bgr(in_yuv_data, 10)

def cvt_yuv444p_to_bgr(in_yuv_data, bitdepth):
  ''' Convert image from yuv444p to bgr image using cv2 color conversion
  
  Params
  ------
    yuv_data: in yuv444p format, 10 bit

  Return
  ------
    a ndarray with dtype uint8, HW3, in BGR format 
  '''
  if bitdepth == 10:
  # to 8 bit
    yuv_data = (in_yuv_data / 4).astype('uint8')
  else:
    yuv_data = in_yuv_data.astype('uint8')

  # using YCrCb to do the color conversion
  ycrcb_data = yuv_data[:,:,[0,2,1]] # HW3, in Y,Cr,Cb format
  img_bgr = cv2.cvtColor(ycrcb_data, cv2.COLOR_YCrCb2BGR)
  return img_bgr

def cvt_yuv444p_to_rgb(in_yuv_data, bitdepth):
  ''' Convert image from yuv444p to rgb image using cv2 color conversion
  
  Params
  ------
    yuv_data: in yuv444p format, 10 bit

  Return
  ------
    a ndarray with dtype uint8, HW3, in RGB format 
  '''
  if bitdepth == 10:
  # to 8 bit
    yuv_data = (in_yuv_data / 4).astype('uint8')
  else:
    yuv_data = in_yuv_data.astype('uint8')

  # using YCrCb to do the color conversion
  ycrcb_data = yuv_data[:,:,[0,2,1]] # HW3, in Y,Cr,Cb format
  img_rgb = cv2.cvtColor(ycrcb_data, cv2.COLOR_YCrCb2RGB )
  return img_rgb

def to_limited_range(yuv_data):
  '''convert full range to limited range. Y: 16:235, U,V: 16:240
  '''
  tmp_data = yuv_data.astype(float)
  tmp_data[:,:,0] = tmp_data[:,:,0]/255 * (235-16) + 16
  tmp_data[:,:,[1,2]] = tmp_data[:,:,[1,2]]/255 * (240-16) + 16
  return tmp_data.astype('uint8')

def limited_to_full(yuv_data):
  '''convert 10bit YUV from limited range to full range
  '''
  tmp_data = yuv_data.astype(float)
  L_YUV=64;H_Y=940;H_UV=960
  tmp_data[:,:,0] = (tmp_data[:,:,0] - L_YUV)/(H_Y - L_YUV) * 1023
  tmp_data[:,:,[1,2]] = (tmp_data[:,:,[1,2]] - L_YUV)/(H_UV - L_YUV) * 1023
  return np.clip(tmp_data, 0, 1023).astype('uint16')
  
def cvt_bgr_to_yuv420p10b(bgr_data):
  ''' Convert image from bgr to yuv420p10b format
  
  Params
  ------
    bgr_data: ndarray in HW3 BGR format with dtype uint8

  Return
  ------
    Y, U, V: ndarray, with dtype uint16. Y component has shape HW3. U and V 
    components have shape (H//2, W//2, 3)
  '''
  ycrcb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2YCrCb)
  #ycrcb_data = to_limited_range(ycrcb_data)
  yuv_data = ycrcb_data[:,:,[0,2,1]]
  return cvt_yuv444p_to_yuv420p10b(yuv_data)

def png_to_yuv420p_ffmpeg(in_fnames, out_fname, bitdepth, item):
  ''' Convert video in png files into YUV420p8b video file using ffmpeg

  Params
  ------
    in_fnames: input png files
    out_fname: output yuv file
  ''' 

  with tempfile.TemporaryDirectory() as temp_dir: 
    for idx, fname in enumerate(in_fnames):
      # pad to make the size even
      padded_img_fname = os.path.join(temp_dir, f'padded_{idx:06d}.png')
      cmd = [
        item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', '1',
        '-i', fname,
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        padded_img_fname]
      err = utils.start_process_expect_returncode_0(cmd, wait=True)
      assert err==0, 'Generating padded png file failed'

    # PNG to YUV 420 8bit
    if bitdepth==8:
      pixfmt = 'yuv420p'
    elif bitdepth==10:
      pixfmt = 'yuv420p10le'
    else: 
      assert False, "Bitdepth is not supported"

    cmd = [
      item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
      '-threads', '1',
      '-i', os.path.join(temp_dir, 'padded_%06d.png'),
      '-f', 'rawvideo', 
      '-pix_fmt', pixfmt,
      '-dst_range', '1',
      out_fname] 

    err = utils.start_process_expect_returncode_0(cmd, wait=True)
    assert err==0, 'Generating sequence in YUV format failed'
 

def png_to_yuv420p10b(in_fnames, out_fname):
  ''' Convert video in png files into YUV420p10b video file

  Params
  ------
    in_fnames: input png files
    out_fname: output yuv file
  ''' 
  with open(out_fname, 'wb') as of:
    for fname in in_fnames:
      img_bgr = cv2.imread(fname) #BGR
      # pad to make the size even
      H,W,C = img_bgr.shape
      img_bgr = cv2.copyMakeBorder(img_bgr, 0, math.ceil(H/2)*2-H, 0, math.ceil(W/2)*2-W, cv2.BORDER_CONSTANT, value=0)
      yy, uu, vv = cvt_bgr_to_yuv420p10b(img_bgr)
      of.write(yy.tobytes())
      of.write(uu.tobytes())
      of.write(vv.tobytes())
 
def yuv420p10b_to_png(in_fname, resolution, out_fname):
  yuv420p_to_png(in_fname, resolution, 10, out_fname)

def yuv420p_to_png(in_fname, resolution, bitdepth, out_fname):
  # load frame
  frame_idx = 0
  if ':' in in_fname:
    yuv_fname, frame_idx = in_fname.split(':')
    frame_idx = int(frame_idx)
  else:
    yuv_fname = in_fname

  H, W, C = resolution
  img_yuv = get_frame_from_raw_video(yuv_fname, frame_idx, 
            W = W, H = H,
            chroma_format = '420',
            bit_depth = bitdepth)
  #img_yuv = limited_to_full(img_yuv)
  img_bgr = cvt_yuv444p_to_bgr(img_yuv, bitdepth)
  cv2.imwrite(out_fname, img_bgr)

def yuv420p10b_to_png_ffmpeg(in_fname, resolution, out_fname, item):
  yuv420p_to_png_ffmpeg(in_fname, resolution, 10, out_fname, item)

def yuv420p_to_png_ffmpeg(in_fname, resolution, bitdepth, out_fname, item):
  '''Convert YUV 420p 10-bit to PNG format using ffmpeg
  '''
  # YUV 420p 10-bit to PNG format using ffmpeg
  idx = 0
  if ':' in in_fname:
    in_fname, idx = in_fname.split(':')
    idx = int(idx)

  if bitdepth == 10:
    fmt = "yuv420p10le"
  else:
    fmt = "yuv420p"

  H,W,C = resolution
  cmd = [
    item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
    '-threads', '1',
    '-f', 'rawvideo', 
    '-pix_fmt', fmt,
    '-s', f"{W}x{H}",
    '-src_range', '1',
    '-i', in_fname,
    '-vf', f"select=eq(n\,{idx})",
    '-vframes', '1',
    '-pix_fmt', 'rgb24',
    out_fname] 

  err = utils.start_process_expect_returncode_0(cmd, wait=True)
  assert err==0, 'Converting from YUV to PNG failed'
 
# convert 10bit to 8 bit
def yuv_10bit_to_8bit(fname, ofname):
  ''' Convert YUV data from 10bit ti 8bit

  Params
  ------
    fname: input file name for the YUV data in 10bit
    ofname: output file name for the YUV data in 8 bit
  '''
  with open(fname, 'rb') as f:
    data = f.read()
  data = np.frombuffer(data, 'uint16')
  data = (data / 4).astype('uint8')
  with open(ofname, 'wb') as f:
    f.write(data.tobytes())
    
# convert 8bit to 10bit
def yuv_8bit_to_10bit(fname, ofname):
  ''' Convert YUV data from 8bit to 10bit

  Params
  ------
    fname: input file name for the YUV data in 8bit
    ofname: output file name for the YUV data in 10bit
  '''
  with open(fname, 'rb') as f:
    data = f.read()
  data = np.frombuffer(data, 'uint8')
  data = (data.astype('uint16') * 4).astype('uint16')
  with open(ofname, 'wb') as f:
    f.write(data.tobytes())

# convert YUV data into yuv420p10le
# bitdepth: 8 or 10
# chroma_format: 420 or 444
def yuv_to_yuv420p10le(in_fname, out_fname, width, height, bitdepth, chroma_format, item):
  in_format = 'yuv'
  in_format += chroma_format + 'p'
  if bitdepth == 10:
    in_format += "10le"

  # no conversion if the format is already correct
  if in_format == 'yuv420p10le':
     io_utils.enforce_symlink(in_fname, out_fname)
     return

  cmd = [
    item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
    '-threads', '1',
    '-f', 'rawvideo', 
    '-pix_fmt', in_format,
    '-s', f"{width}x{height}",
    '-i', in_fname,
    '-f', 'rawvideo',
    '-pix_fmt', 'yuv420p10le',
    out_fname] 

  err = utils.start_process_expect_returncode_0(cmd, wait=True)
  assert err==0, 'Converting from YUV to yuv420p10le failed'
 

# concatenate YUV files, overalpped frames are removed
def concat_yuvs(fnames, width, height, bitdepth, num_overlap_frames, out_fname):
  data = bytearray()
  # get frame size
  frame_size = get_frame_size_from_yuv(
    width,
    height,
    InputChromaFormat='420', 
    InputBitDepth=bitdepth)
  
  for fname in fnames:    
    frame_idx = 0
    num_frames = -1
    if ':' in fname: 
      fname, frame_idx = fname.split(':')
      frame_idx = int(frame_idx)
      num_frames = 1

    with open(fname, 'rb') as f:
      d1 = f.read()
      if len(data) > 0:
        if num_frames == -1:
          d1 = d1[(frame_size*(num_overlap_frames+frame_idx)):]
        else:
          d1 = d1[(frame_size*(num_overlap_frames+frame_idx)):frame_size*(num_overlap_frames+frame_idx+1)]
      data += d1
  with open(out_fname, 'wb') as f:
    f.write(data)

# resize video
# return the new resolution in H,W,C format
def resize_image(fname, out_fname, scale_factor=0.75):
  img = cv2.imread(fname)
  H,W,C = img.shape
  out_img = cv2.resize(img, 
    (int(np.round(W*scale_factor)), int(np.round(H*scale_factor))), 
    interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(out_fname, out_img)
  return out_img.shape

# pad image
def pad_image(fname, out_fname, pad_h, pad_w):
  img = cv2.imread(fname)
  H,W,C = img.shape
  out_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
  cv2.imwrite(out_fname, out_img)
  return out_img.shape

# unpad an image
def unpad_image(fname, out_fname, pad_h, pad_w):
  img = cv2.imread(fname) #HWC
  H,W,C = img.shape
  out_img = img[:H-pad_h,:W-pad_w,:]
  cv2.imwrite(out_fname, out_img)
  return out_img.shape

def pad_image_even_ffmpeg(fname, out_fname, item):
  cmd = [
        item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', '1',
        '-i', fname,
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        out_fname]

  err = utils.start_process_expect_returncode_0(cmd, wait=True)
  assert err==0, 'Generating padded png file failed'

# unpad an image
def unpad_image_ffmpeg(fname, out_fname, width, height, item):
  cmd = [
    item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
    '-threads', '1',
    '-i', fname,
    '-vf', f"crop={width}:{height}",
    out_fname,
  ]
  err = utils.start_process_expect_returncode_0(cmd, wait=True)
  assert err==0, 'Generating padded png file failed'

# Resize YUV video
def scale_video_yuv_ffmpeg(in_fname, out_fname, scale, height, width, in_bitdepth=-1, item=None):
  """
    Scale an yuv video with ffmpeg
    NOTE: This function does not check for the vadility of the input or output resolution.
  """
  C = 3
  if in_bitdepth == 10:
    fmt = "yuv420p10le"
  elif in_bitdepth == 8:
    fmt = "yuv420p"
  else:
    raise ValueError(f"Wrong value given for input bitdepth: {in_bitdepth}")

  cmd = [
    item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
    '-threads', '1',
    '-f', 'rawvideo', 
    '-pix_fmt', fmt,
    '-s', f"{width}x{height}",
    '-i', in_fname,
    '-vf', f"scale=iw*{scale}:ih*{scale}:flags=bilinear,format={fmt}",
    '-pix_fmt', fmt,
    out_fname] 

  err = utils.start_process_expect_returncode_0(cmd, wait=True)
  assert err==0, f'Resizing failed with file {in_fname}.'
  return C,int(height*scale),int(width*scale)

###############################################
#
def psnr(f1, f2):
  ''' Calculate psnr from two images

  Params
  ------
    f1, f2: two input image data to compute the PSNR value. Dtype as follows
      uint8: 8bit data
      uint16: 10bit data
      float, float64, float32: float format in range [-1, 1]

  Return
  ------
    PSNR value
  '''
  if type(f1) == np.ndarray:
    # informat HW3
    max_dict = {'uint8': 255,
                'uint16': 1024,
                'float': 1,
                'float64': 1,
                'float32': 1,
               }

    # convert float to 0-1
    if f1.dtype == float:
       f1 = (f1 + 1)/2

    if f2.dtype == float:
       f2 = (f2 + 1)/2

    f1 = f1.astype(float) / max_dict[str(f1.dtype)]
    f2 = f2.astype(float) / max_dict[str(f2.dtype)]
    mse = np.mean((f1-f2)**2)
    psnr = 10 * np.log10( 1 / (mse + 1e-12))
    return psnr


