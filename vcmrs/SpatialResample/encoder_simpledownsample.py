# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import cv2
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from . import simpledownsample_data
from vcmrs.Utils.io_utils import enforce_symlink 

class SimpleDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)
    self.scale_factor_mapping = simpledownsample_data.scale_factor_id_mapping

  def process(self, input_fname, output_fname, item, ctx):
    resized = False
    if not os.path.exists(input_fname):
      raise FileNotFoundError(f"Input file {input_fname} is not found")
    
    if item._is_video and not item._is_yuv_video: # video as directory
      os.makedirs(output_fname, exist_ok=True)
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
        resized = self._resize_frame(fname, output_frame_fname, item, ctx)
    elif os.path.isfile(input_fname): # YUV file or image data
      # image data
      resized = self._resize_frame(input_fname, output_fname, item, ctx)
    else:
      raise FileNotFoundError(f"Input {input_fname} is not found.")

    if resized:
      self._set_parameter(item, item.args.OversizedVideoScaleFactor)

      # For illustration of inner codec control signal
      # The signal data can be any python object
      item.add_inner_control_signal('SpatialResample', signal={'data1': 0, 'data2': 2})

  def _resize_frame(self, input_fname, output_fname, item, ctx):
    # get image size
    resized = False

    if item._is_yuv_video :
      H,W = item.args.SourceHeight, item.args.SourceWidth
    else: # frame data
      img = cv2.imread(input_fname)
      H,W,C = img.shape

    if os.path.isfile(output_fname): os.remove(output_fname)
    if max(H, W) > item.args.ResolutionThreshold:
      if item._is_yuv_video:
        C,H,W = data_utils.scale_video_yuv_ffmpeg(input_fname, output_fname, item.args.OversizedVideoScaleFactor, H, W, in_bitdepth=item.args.InputBitDepth, item=item)
      else:
        C,H,W = data_utils.resize_image(input_fname, output_fname, item.args.OversizedVideoScaleFactor)

      item.args.SourceWidth = W
      item.args.SourceHeight = H

      resized = True
    else:
      enforce_symlink(input_fname, output_fname) 
    return resized


  def _set_parameter(self, item, scale_factor):
    # scale factor is a video scale parameter
    # sequence level parameter
    param_data = bytearray([self.scale_factor_mapping[scale_factor]])
    item.add_parameter('SpatialResample', param_data=param_data) 
    pass




