
    
# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import datetime
import shutil
import vcmrs
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink
import numpy as np



class truncation(Component):
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
      vcmrs.log("========================================== truncation start =====================================================")
      if item._is_yuv_video:
        height = item.args.SourceHeight
        width = item.args.SourceWidth
        bytes_per_pixel = 2 if item.args.InputBitDepth > 8 else 1

        # Determine sizes based on format
        if item.args.InputChromaFormat == "420":
            y_size = width * height
            uv_size = (width // 2) * (height // 2)
            uv_shape = (height // 2, width // 2)
        elif item.args.InputChromaFormat == "422":
            y_size = width * height
            uv_size = (width // 2) * height
            uv_shape = (height, width // 2)
        elif item.args.InputChromaFormat == "444":
            y_size = width * height
            uv_size = width * height
            uv_shape = (height, width)
        else:
            raise ValueError("Unsupported chroma format: {}".format(item.args.InputChromaFormat))
          
        input_file = open(input_fname, 'rb')
        output_file = open(output_fname, 'wb')

        while True:
          # read Y, U, V
          y_buffer = input_file.read(y_size * bytes_per_pixel)
          if len(y_buffer) == 0:  # check if at the end of the file
              break
          y = np.frombuffer(y_buffer, dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape((height, width))  
          u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)
          v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape)

          y = y.copy()
          # right shift 1
          y = np.right_shift(y, 1) 

          output_file.write(y.astype(np.uint16 if bytes_per_pixel == 2 else np.uint8).tobytes())
          output_file.write(u.tobytes())
          output_file.write(v.tobytes())
          
        # close file
        input_file.close()
        output_file.close()
        bit_depth_shift_flag = 1 if item.args.OriginalSourceHeight < item.args.BitTruncationRestorationHeightThreshold and item.args.OriginalSourceWidth < item.args.BitTruncationRestorationWidthThreshold else 0
        # bit_depth_shift_flag=1 means: shift left (restore) at the decoder
        
        bit_depth_shift_luma = 1 if bit_depth_shift_flag else 0
        bit_depth_shift_chroma = 0 
        self._set_parameter(item, bit_depth_shift_flag, bit_depth_shift_luma, bit_depth_shift_chroma)
        vcmrs.log("================================= complete truncation =================================")
      
      #non-yuv files
      else:
        enforce_symlink(input_fname, output_fname)
    


  def _set_parameter(self, item, bit_depth_shift_flag, bit_depth_shift_luma = 0, bit_depth_shift_chroma = 0):
    # sequence level parameter
    #default scale: 1 byte, framestoberestore: 2 bytes
    param_data = bytearray(3)
    param_data[0] = bit_depth_shift_flag
    param_data[1] = bit_depth_shift_luma
    param_data[2] = bit_depth_shift_chroma
    item.add_parameter('BitDepthTruncation', param_data=param_data)
    pass

