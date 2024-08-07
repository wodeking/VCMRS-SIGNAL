# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import cv2
import datetime
import time
import numpy as np
import shutil
import vcmrs
from vcmrs.Utils.io_utils import enforce_symlink, rm_file_dir
from vcmrs.Utils.component import Component
from . import resample_data
from .Interpolation import Interpolation, Interpolation_frame_by_frame, _UpResampler_, VideoCaptureYUV, cal_md5
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
    starttime = time.time()
    
    PHTemporalRatioIndexes = None 
    VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, default_scale_factor = self._get_parameter(item)

    default_idx = resample_data.scale_factor_id_mapping[default_scale_factor]
    ratio_change_flag = [int(i) for i in PHTemporalChangedFlags]
    remain = TemporalRemain
    scale_factor = default_scale_factor

    # make FramesToBeRecon
    FramesToBeRecon = 0
    tmp_scale_factor = default_scale_factor
    for chg_flag in ratio_change_flag[:-1]:
      if chg_flag != 0:
        tmp_scale_factor = resample_data.scale_factor_id_mapping[int(chg_flag)]
      FramesToBeRecon += tmp_scale_factor
    FramesToBeRecon += (TemporalRemain + 1)

    if PHTemporalRatioIndexes is not None:
        all_scale_idx = PHTemporalRatioIndexes
    else:
        all_scale_idx = [default_idx]
    
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
      # Interpolation frame by frame  
      # convert yuv420p10le to yuv420p
      timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
      pngpath = os.path.join(os.path.abspath(item.args.working_dir), os.path.basename(input_fname), 'tmp', f'png_{timestamp}')
      os.makedirs(pngpath)

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

      # create output folder
      output_fname_png = output_fname[:-4]
      if not os.path.exists(output_fname_png):
          os.makedirs(output_fname_png)

      frameBuffer = None
      videogen = []
      for f in sorted(os.listdir(pngpath)):
          if "png" in f:
              videogen.append(os.path.join(pngpath, f))

      # build resampler
      upresampler = _UpResampler_()
      savepath = output_fname_png
      upresampler.setAttr(None, 4, savepath, 0, scale_factor)
      if os.path.exists(savepath):
          shutil.rmtree(savepath)
      os.makedirs(savepath)

      count = 0
      temp_I0 = None
      while True:
        ret = (len(videogen) != 0)
        if ret:
          framep = videogen.pop(0)
          frame = cv2.imread(framep, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()

          if frameBuffer is None:
            frameBuffer = frame
            continue
          else:
            bchanged = ratio_change_flag.pop(0)

            if bchanged:
                scale_factor = resample_data.scale_factor_id_mapping[all_scale_idx.pop()]
            
            I0, I1 = upresampler.preprocess(frameBuffer, frame)

            if temp_I0 is not None:
              I0 = temp_I0 

            ssim = upresampler.cal_ssim(I0, I1)

            if ssim > 0.996 :
              if len(videogen) != 0:
                framep = videogen[0]
                frame2 = cv2.imread(framep, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
              else:
                frame2 = frame
              _, I2 = upresampler.preprocess(frameBuffer, frame2)
              I1 = upresampler.model.inference(I0, I2, 1.0)              
              frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W]
              temp_I1 = I1
              update_flag = True
            else:
              temp_I1 = None
              update_flag = False
            
            upresampler.saveImg(frameBuffer)

            if scale_factor != 1:
              Interpolation_frame_by_frame(upresampler, frameBuffer, frame, scale_factor, (temp_I0, temp_I1))

            if update_flag:
              temp_I0 = I1
            else:
              temp_I0 = None


            count += scale_factor
            frameBuffer = frame

        else:
          break
      
      # deal with tail 
      # remain = FramesToBeRecon - count - 1
      tail = remain + 1
      if tail != 0:
        Interpolation_frame_by_frame(upresampler, frameBuffer, frame, scale_factor, (None, temp_I1), tail=tail)  
      
      # convert pngs to yuv
      out_frame_fnames = os.path.join(output_fname_png, 'frame_%06d.png')
      vcmrs.log(f'Interpolate {len(os.listdir(output_fname_png))} pictures')
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
    
    durationtime = time.time() - starttime
    vcmrs.log(f'Temporal decoding completed in {durationtime} seconds')

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

  def _get_parameter(self, item):
    # sequence level parameter
    # phlength: 2 byte, VCMEnabled: 1 byte, TemporalEnabled: 1 byte, PHTemporalChangedFlags: (length of self -1 ) // 8 + 1 byte, TemporalRemain: 1 byte
    phlength = None
    VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, extRate = None, None, [], None, None
    param_data = item.get_parameter('TemporalResample')
    if param_data is not None:
      phlength = (param_data[0] << 8) + param_data[1]
      bytelength = (phlength - 1) // 8 + 1
      assert len(param_data) == (bytelength + 6), f'received parameter data is not correct: {param_data}'
      VCMEnabled = param_data[2]
      TemporalEnabled = param_data[3]
      for i in range(bytelength):
        res = "00000000" + bin(param_data[i+4])[2:]

        for j in res[-8:]:
          PHTemporalChangedFlags.append(j)
      PHTemporalChangedFlags = PHTemporalChangedFlags[:phlength]
      TemporalRemain = param_data[4 + bytelength]
      
      extRate = param_data[5 + bytelength]
      vcmrs.log(f'phlength:{phlength}, VCMEnabled:{VCMEnabled}, TemporalEnabled:{TemporalEnabled}, PHTemporalChangedFlags: {PHTemporalChangedFlags}, TemporalRemain:{TemporalRemain}, encoder extRate:{extRate}')
    return VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, extRate
