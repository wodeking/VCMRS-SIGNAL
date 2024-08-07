# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink 

from . import adaptivedownsample_data

class AdaptiveDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)
    self.id_scale_factor_mapping = adaptivedownsample_data.id_scale_factor_mapping

  def process(self, input_fname, output_fname, item, ctx):

    # the default implementation is a bypass component
    if item._is_dir_video:
      # video dir data
      os.makedirs(output_fname, exist_ok=True)
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")

    elif os.path.isfile(input_fname): 
      # image or YUV data
      os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    else:
      raise FileNotFoundError(f"Input {input_fname} is not found.")

    if os.path.isfile(output_fname): os.remove(output_fname)
    enforce_symlink(input_fname, output_fname)