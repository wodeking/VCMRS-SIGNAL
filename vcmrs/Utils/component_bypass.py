# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink

class Bypass(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
    # the default implementation is a bypass component

    #if item._is_dir_video:
    if os.path.isdir(input_fname): #item._is_dir_video:
      # video data in a directory
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
        if os.path.isfile(output_frame_fname): os.remove(output_frame_fname)
        os.makedirs(os.path.dirname(output_frame_fname), exist_ok=True)
        enforce_symlink(fname, output_frame_fname)
    else:
      # image data or video in yuv format
      if os.path.isfile(output_fname): os.remove(output_fname)
      os.makedirs(os.path.dirname(output_fname), exist_ok=True)
      enforce_symlink(input_fname, output_fname)


