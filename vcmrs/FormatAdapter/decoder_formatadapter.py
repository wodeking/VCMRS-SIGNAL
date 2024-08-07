# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import shutil
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink

class formatadapter(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
    if os.path.isdir(input_fname):
      if getattr(item.args, "directory_as_video", False) or \
        getattr(item.args, "output_video_format", "YUV")=='PNG':
        # by pass
        self._by_pass(input_fname, output_fname)
      else:
        # convert PNG to yuv420p10le
        # video dir data
        os.makedirs(output_fname, exist_ok=True)
        fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
        data_utils.png_to_yuv420p_ffmpeg(fnames, output_fname, bitdepth=10, item=item)

    elif os.path.isfile(input_fname): 
      # by pass these formats, YUV should be in 420 10bit already
      # image data or video in yuv format
      self._by_pass(input_fname, output_fname)

    else:
      raise FileNotFoundError(f"Input {input_fname} is not found.")

  def _by_pass(self, input_fname, output_fname):
    if os.path.isfile(output_fname): os.remove(output_fname)
    if os.path.islink(output_fname): os.unlink(output_fname)
    if os.path.isdir(output_fname): shutil.rmtree(output_fname)
    #os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    enforce_symlink(input_fname, output_fname)


