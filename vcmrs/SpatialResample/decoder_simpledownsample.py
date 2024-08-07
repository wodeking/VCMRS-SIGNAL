# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink 

from . import simpledownsample_data

class SimpleDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)
    self.id_scale_factor_mapping = simpledownsample_data.id_scale_factor_mapping

  def process(self, input_fname, output_fname, item, ctx):
    scale_factor =self._get_parameter(item)

    # the default implementation is a bypass component
    if item._is_dir_video:
      # video dir data
      os.makedirs(output_fname, exist_ok=True)
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
        self._resize_frame(fname, output_frame_fname, scale_factor, item)

    elif os.path.isfile(input_fname): 
      # image or YUV data
      os.makedirs(os.path.dirname(output_fname), exist_ok=True)
      self._resize_frame(input_fname, output_fname, scale_factor, item)
    else:
      raise FileNotFoundError(f"Input {input_fname} is not found.")

  def _resize_frame(self, input_fname, output_fname, scale_factor, item=None):
     if os.path.isfile(output_fname): os.remove(output_fname)
     if scale_factor == 1:
       enforce_symlink(input_fname, output_fname)
     else:
       if os.path.splitext(input_fname)[1].lower() == '.yuv':
         H,W,C = item.video_info.resolution
         C,H,W = data_utils.scale_video_yuv_ffmpeg(input_fname, output_fname, 1/scale_factor, H, W, in_bitdepth=10, item=item)
       else:
         C,H,W = data_utils.resize_image(input_fname, output_fname, 1/scale_factor)
       item.video_info.resolution = (H,W,C)

  def _get_parameter(self, item):
    # sequence level parameter
    scale_factor = 1
    param_data = item.get_parameter('SpatialResample')
    if param_data is not None: 
      assert len(param_data)==1, f'received parameter data is not not correct: {param_data}'
      scale_factor = self.id_scale_factor_mapping[param_data[0]]
    return scale_factor


