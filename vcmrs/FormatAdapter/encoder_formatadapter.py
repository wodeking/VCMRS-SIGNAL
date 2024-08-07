# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import shutil
import glob
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink

'''
  This component convert input data into an internal data format. There are three types of input 
  fomrat: 
    - YUV_video: Video in YUV format
    - PNG_video: Video frames in a directory in jpg/png format
    - Image: RGB Image in jpg/ng format

  The output format of this plugin is
    - YUV_video: YUV420p10le  
    - PNG_video: unchanged
    - Image: unchanged
'''
class formatadapter(Component): 
  # initialize the plugin
  def __init__(self, ctx):
    super().__init__(ctx)

  # process an item
  def process(self, input_fname, output_fname, item, ctx):
    if item._is_video and item._is_yuv_video:
      # convert to YUV420ple
      #def yuv_to_yuv420p10le(in_fname, out_fname, width, height, bitdepth, chroma_format):
      data_utils.yuv_to_yuv420p10le(input_fname, output_fname, 
        width=item.args.SourceWidth,
        height=item.args.SourceHeight,
        bitdepth=item.args.InputBitDepth,
        chroma_format=item.args.InputChromaFormat,
        item=item)
      # set the corresponding parameters after the conversion
      item.args.InputBitDepth=10
      item.args.InputChromaFormat='420'
    else:
      if os.path.isfile(output_fname): os.remove(output_fname)
      if os.path.islink(output_fname): os.unlink(output_fname)
      if os.path.isdir(output_fname): shutil.rmtree(output_fname)
      enforce_symlink(input_fname, output_fname)


