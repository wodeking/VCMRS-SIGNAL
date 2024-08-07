# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import cv2
import datetime
import numpy as np
import shutil
import vcmrs
from vcmrs.Utils.io_utils import enforce_symlink, rm_file_dir
from vcmrs.Utils.component import Component
from . import resample_data
from .Interpolation import Interpolation
from vcmrs.Utils import utils

class resample(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    self.id_scale_factor_mapping = resample_data.id_scale_factor_mapping

  def _updateExtArgs(self, item, num_frames, extRate):
  
    item.FrameRateRelativeVsInnerCodec = item.FrameRateRelativeVsInnerCodec * extRate
  
    pre_temporal_num_frames = item.video_info.num_frames
    item.video_info.num_frames = num_frames

    # AllIntra Mode
    if len(item.video_info.intra_indices)>1 and (item.video_info.intra_indices[1] - item.video_info.intra_indices[0])==1:
      item.video_info.intra_indices = list(range(num_frames))
    else:
      for k in range(len(item.video_info.intra_indices)):
        item.video_info.intra_indices[k] *= extRate

    qps = np.array([10000]*num_frames)
    newQP = 10000
    for i in range(pre_temporal_num_frames-1):
      qps[extRate*i] = item.video_info.frame_qps[i]
      if item.video_info.frame_qps[i]<0 and item.video_info.frame_qps[i+1]<0:
        newQP = item.video_info.frame_qps[i]
      elif min(item.video_info.frame_qps[i],item.video_info.frame_qps[i+1])<0:
        newQP = max(item.video_info.frame_qps[i], item.video_info.frame_qps[i+1])
      else:
        newQP = round((item.video_info.frame_qps[i]+item.video_info.frame_qps[i + 1])/2)
      for j in range(extRate*i+1,extRate*(i+1)):
        qps[j] = newQP
    qps[extRate*(pre_temporal_num_frames-1)] = item.video_info.frame_qps[pre_temporal_num_frames-1]
    if (num_frames-1)%extRate!=0:
      qps[(extRate*(pre_temporal_num_frames-1)+1):]=qps[extRate*(pre_temporal_num_frames-1)]
    item.video_info.frame_qps = qps.tolist() #convert to native type

  def process(self, input_fname, output_fname, item, ctx):
    vcmrs.log('######decode temporal process###########')
    VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain = self._get_parameter(item)
    
    if FramesToBeRecon is None:
      # scale_factor and FramesToBeRecon will be None when the encoder doesn't set temporal parameters
      #link the output to input directly
      os.makedirs(os.path.dirname(output_fname), exist_ok=True)
      rm_file_dir(output_fname)
      enforce_symlink(input_fname, output_fname) 
      return

    if item._is_dir_video:
      # input format ： directory
      pngpath = input_fname
      output_fname_png = output_fname
      bFilter = True
      if item.args.InnerCodec == 'VTM':
        bFilter = False
      self._resize_frame(pngpath, output_fname_png, scale_factor, FramesToBeRecon, bFilter)
      self._updateExtArgs(item, FramesToBeRecon, scale_factor)
    elif item._is_yuv_video:
      # convert yuv to pngs
      timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
      pngpath = os.path.join(os.path.abspath(item.args.working_dir), os.path.basename(input_fname), 'tmp', f'png_{timestamp}')
      os.makedirs(pngpath, exist_ok=True)
      H,W,C = item.video_info.resolution
      tmpname = os.path.join(pngpath, "frame_%06d.png")

      cmd = [
        item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', '1',
        '-f', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'yuv420p10le',
        '-i', input_fname,
        '-vsync', '0',
        '-y',
        '-pix_fmt', 'rgb24', 
        tmpname] 

      err = utils.start_process_expect_returncode_0(cmd, wait=True)
      assert err==0, "Generating sequence in YUV format failed."

      output_fname_png = output_fname[:-4]
      os.makedirs(output_fname_png, exist_ok=True)
      # do the filter for NNVVC mode
      bFilter = True
      if item.args.InnerCodec == 'VTM':
        bFilter = False
      self._resize_frame(pngpath, output_fname_png, scale_factor, FramesToBeRecon, bFilter)
      # convert pngs to yuv
      out_frame_fnames = os.path.join(output_fname_png, 'frame_%06d.png')
      cmd = [
        item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', '1',
        '-i', out_frame_fnames,
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p10le',
        output_fname] 
      err = utils.start_process_expect_returncode_0(cmd, wait=True)
      assert err==0, "Generating sequence in YUV format failed."
      self._updateExtArgs(item, FramesToBeRecon, scale_factor)
    else:
      assert False, f"Input file {input_fname} is not found"

  def _resize_frame(self, input_fname, output_fname, scale_factor, FramesToBeRecon, bFilter):
    #default input_fname and output_fname are both directory
    rm_file_dir(output_fname)
    if scale_factor == 1:
      rm_file_dir(output_fname)
      os.makedirs(output_fname, exist_ok=True)
      for file in os.listdir(input_fname):
        idx = int(os.path.basename(file).split('.')[0].split('_')[-1])
        srcfile = os.path.join(input_fname, file)
        desfile = os.path.join(output_fname, 'frame_%06d.png'%(idx)) # %07d.png
        rm_file_dir(desfile)
        enforce_symlink(srcfile, desfile) 
    else:
      if bFilter:
        # do the filter before temporal up-resample
        first = sorted(os.listdir(input_fname))[0]
        srcfile = os.path.join(input_fname, first)
        img = cv2.imread(srcfile)
        kernel = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
        img3 = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(srcfile, img3)
        # do the temporal up-resample
      Interpolation(input_fname, output_fname, scale_factor, FramesToBeRecon)

  def _get_parameter_bk(self, item):
    # sequence level parameter
    scale_factor = None
    FramesToBeRecon = None
    param_data = item.get_parameter('TemporalResample')
    if param_data is not None:
      assert len(param_data) == 3, f'received parameter data is not correct: {param_data}'
      scale_factor = self.id_scale_factor_mapping[param_data[0]]
      FramesToBeRecon = (param_data[1] << 8) + param_data[2]
      vcmrs.log(f'decoder scale_factor:{scale_factor}, FramesToBeRecon:{FramesToBeRecon}')
    return scale_factor, FramesToBeRecon

  def _get_parameter(self, item):
    # sequence level parameter
    # total length: 2 byte, VCMEnabled: 1 byte, TemporalEnabled: 1 byte, PHTemporalChangedFlags: (length of self -1 ) // 8 + 1 byte, TemporalRemain: 1 byte
    phlength = None
    VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain = None, None, [], None
    param_data = item.get_parameter('TemporalResample')
    if param_data is not None:
      phlength = (param_data[0] << 8) + param_data[1]
      assert len(param_data) == (phlength + 5), f'received parameter data is not correct: {param_data}'
      VCMEnabled = param_data[2]
      TemporalEnabled = param_data[3]
      for i in range(phlength):
        res = "00000000" + bin(param_data[i+4])[2:]
        for j in res[-8:]:
          PHTemporalChangedFlags.append(j)
      TemporalRemain = param_data[4 + phlength]
      vcmrs.log(f'VCMEnabled:{VCMEnabled}, TemporalEnabled:{TemporalEnabled}, PHTemporalChangedFlags: {PHTemporalChangedFlags}, TemporalRemain:{TemporalRemain}')
    return VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain
