# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import datetime
import shutil
import vcmrs
from vcmrs.Utils.io_utils import enforce_symlink 
from vcmrs.Utils.component import Component
from vcmrs.InnerCodec.Utils import video_utils
from . import resample_data
from vcmrs.Utils import utils

def rm_file_dir(fname):
    if os.path.exists(fname):
        if os.path.isdir(fname):shutil.rmtree(fname)
        else:os.remove(fname)

def yuvTopngs(input_fname, item, pngdir, num_frames, video_info):
    os.makedirs(pngdir, exist_ok=True)
    start = item.args.FrameSkip
    end = start + num_frames - 1
    pixfmt = video_utils.get_ffmpeg_pix_fmt(video_info)
    tmpname = os.path.join(pngdir, "frame_%06d.png")
    cmd = [
      item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
      '-threads', '1',
      '-f', 'rawvideo',
      '-s', f'{item.args.SourceWidth}x{item.args.SourceHeight}',
      '-pix_fmt', pixfmt,
      '-i', input_fname,
      '-vf' ,f'select=between(n\,{start}\,{end})',
      '-vsync', '0',
      '-y',
      '-pix_fmt', 'rgb24', 
      tmpname] # %06d.png

    err = utils.start_process_expect_returncode_0(cmd, wait=True)
    assert err == 0, "convert yuv to png failed."

class resample(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    self.scale_factor_mapping = resample_data.scale_factor_id_mapping

  def _updateExtArgs(self, item, num_frames):
      item.args.FrameSkip = 0
      item.args.FramesToBeEncoded = num_frames
      if item.IntraPeriod > 1:
          item.IntraPeriod = int(item.IntraPeriod / item.args.TemporalScale)
      item.FrameRateRelativeVsInput = item.FrameRateRelativeVsInput / item.args.TemporalScale
      vcmrs.log(f'item.args.FramesToBeEncoded:{item.args.FramesToBeEncoded}')

  def _ext_files(self, inputdir, outputdir, extRate, item, video_info, yuvFile=False):
      if not os.path.isdir(inputdir):
          assert False, "Input file should be directory!"
      fnames = sorted(glob.glob(os.path.join(inputdir, '*.png')))
      rm_file_dir(outputdir)

      # if frame postfix from 0, idx will add 1
      idx_start = int(os.path.basename(fnames[0]).split('.')[0].split('_')[-1])
      supple = 1-idx_start
      ext_num_frames = 0
      for fname in fnames:
          idx = int(os.path.basename(fname).split('.')[0].split('_')[-1])
          idx = idx + supple
          if (idx - 1) % extRate == 0:
              new_idx = int((idx - 1) / extRate)
              os.makedirs(outputdir, exist_ok=True)
              output_frame_fname = os.path.join(outputdir, f"frame_{new_idx:06d}.png")
              rm_file_dir(output_frame_fname)
              enforce_symlink(fname, output_frame_fname) 
              ext_num_frames+=1
      if yuvFile:
          tmpdir = os.path.splitext(outputdir)[0]
          rm_file_dir(tmpdir)
          os.renames(outputdir, tmpdir)
          pixfmt = video_utils.get_ffmpeg_pix_fmt(video_info)
          out_frame_fnames = os.path.join(tmpdir, 'frame_%06d.png')
          cmd = [
            item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-threads', '1',
            '-i', out_frame_fnames,
            '-f', 'rawvideo',
            '-pix_fmt', pixfmt,
            outputdir] 
          err = utils.start_process_expect_returncode_0(cmd, wait=True)
          assert err == 0, "convert png to yuv failed."
      return ext_num_frames

  def process(self, input_fname, output_fname, item, ctx):
    extRate = item.args.TemporalScale
    # for IntraPeriod downsample, down-sampled IntraPeriod should be compatible with GOP length
    # currently, ramdom access GOP configurations support not smaller than 8
    if item.IntraPeriod>1 and int(item.IntraPeriod/item.args.TemporalScale)<8:
        extRate = int(item.IntraPeriod/8)
    video_info = video_utils.get_video_info(input_fname, item.args)
    num_frames = video_info.num_frames - item.args.FrameSkip
    VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain = self.write_cfg(item)
    
    if item.args.FramesToBeEncoded > 0: num_frames = min(num_frames, item.args.FramesToBeEncoded)

    if item._is_dir_video:
        ext_num_frames = self._ext_files(input_fname, output_fname, extRate, item, video_info)
        # self._updateExtArgs(item, num_frames)
        self._updateExtArgs(item, ext_num_frames)
        # self._set_parameter(item, extRate, num_frames)
        self._set_parameter(item, VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, extRate)
    elif item._is_yuv_video:
        # default the output format of temporal resample is the same as the input
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        pngpath = os.path.join(os.path.abspath(item.args.working_dir), os.path.basename(input_fname), 'tmp', f'{item.args.quality}_{timestamp}')
        os.makedirs(pngpath, exist_ok=True)
        yuvTopngs(input_fname, item, pngpath, num_frames, video_info)
        ext_num_frames = self._ext_files(pngpath, output_fname, extRate, item, video_info, yuvFile=True)
        # self.write_cfg(item)

        # self._updateExtArgs(item, num_frames)
        self._updateExtArgs(item, ext_num_frames)
        # self._set_parameter(item, extRate, num_frames)
        self._set_parameter(item, VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, extRate)
    elif os.path.isfile(input_fname) and (os.path.splitext(input_fname)[1].lower() in ['.png', '.jpg', '.jpeg']):
        rm_file_dir(output_fname)
        enforce_symlink(input_fname, output_fname)
    else:
      assert False, f"Input file {input_fname} is not found"


  def write_cfg(self, item):
    # outputfile = os.path.join(item.args.cvc_dir, 'cfg', 'TEMPAR.cfg')
    
    video_working_dir = item.working_dir
    os.makedirs(video_working_dir, exist_ok=True)
    outputfile = os.path.join(video_working_dir, "TEMPAR.CFG")

    # get parameter
    scalefactor = item.args.TemporalScale

    nframes = item.args.FramesToBeEncoded
    TemporalEnabled = 1 if item.args.TemporalResample != 'Bypass' else 0

    ratiolist = [scalefactor for _ in range((nframes-1) // scalefactor)]
    remain = nframes - ((nframes-1) // scalefactor)*scalefactor -1

    prdTemEnabledFlags = []
    tmpscale = -1
    # make prdTemEnabledFlags
    while len(ratiolist) != 0:
        picratio = ratiolist.pop()
        if tmpscale == picratio:
           prdTemEnabledFlags.append(0)
        else:
           prdTemEnabledFlags.append(1)
           tmpscale = picratio
    
    if os.path.exists(outputfile):
       os.remove(outputfile)

    f = open(outputfile, 'w+')

    content = """#======== extentsion =====================
VCMEnabled			: 1
TemporalEnabled			:{}
PHTemporalChangedFlags		:{}
TemporalRemain              :{}""".format(TemporalEnabled, ",".join(["0"] * (len(prdTemEnabledFlags) + 1)), remain) 
    f.write(content)
    f.close()
    
    return 1, TemporalEnabled, "".join(["0"] * (len(prdTemEnabledFlags) + 1)), remain

  def _set_parameter_bk(self, item, extRate, FramesToBeEncoded):
    # sequence level parameter
    #default scale: 1 byte, framestoberestore: 2 bytes
    param_data = bytearray(3)
    param_data[0] = self.scale_factor_mapping[extRate]
    param_data[1] = (FramesToBeEncoded >> 8) & 0xFF
    param_data[2] = FramesToBeEncoded & 0xFF
    item.add_parameter('TemporalResample', param_data=param_data)
    vcmrs.log(f'encoder extRate:{extRate}, FramesToBeRecon:{FramesToBeEncoded}')
    pass

  def _set_parameter(self, item, VCMEnabled, TemporalEnabled, PHTemporalChangedFlags, TemporalRemain, extRate):
    # sequence level parameter
    # phlength: 2 byte, VCMEnabled: 1 byte, TemporalEnabled: 1 byte, PHTemporalChangedFlags: (length of self -1 ) // 8 + 1 byte, TemporalRemain: 1 byte, default scale: 1 byte
    phlength = len(PHTemporalChangedFlags)
    bytelength = (len(PHTemporalChangedFlags) - 1) // 8 + 1
    totallength = 2 + 1 + 1 + bytelength + 1 + 1
    param_data = bytearray(totallength)
    param_data[0] = ( phlength >> 8 ) & 0xFF
    param_data[1] = phlength & 0xFF
    param_data[2] = VCMEnabled & 0xFF    # VCMEnabled
    param_data[3] = TemporalEnabled & 0xFF

    for i in range(bytelength):
        res = "".join(PHTemporalChangedFlags[i*8:(i+1)*8])
        param_data[i + 4] = (int(res, 2) >> 8)
        print(res, param_data[i+4])
    param_data[bytelength + 4] = TemporalRemain
    param_data[bytelength + 5] = extRate
    item.add_parameter('TemporalResample', param_data=param_data)
    vcmrs.log(f'phlength:{phlength}, VCMEnabled:{VCMEnabled}, TemporalEnabled:{TemporalEnabled}, PHTemporalChangedFlags: {PHTemporalChangedFlags}, TemporalRemain:{TemporalRemain}, encoder extRate:{extRate}')

    pass