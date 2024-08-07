# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
# bitstream tools
from types import SimpleNamespace
import numpy as np
import struct
import csv

from vcmrs.Utils import utils
from vcmrs.Utils import data_utils

# generate bitstream 
def gen_image_bitstream( \
    input_fname,
    output_fname,
    lic_bitstream_fname, 
    param=None):

  H,W,C = data_utils.get_img_resolution(input_fname)
  assert H>0 and W>0, 'The width and height of the input image must be greater than 0'
 
  # generate bitstream for image
  with open(lic_bitstream_fname, 'rb') as f:
    lic_bitstream = f.read()

  # generate header
  header = np.uint32(0)

  header += (W & 0x1fff) << 19 # picture_width
  header += (H & 0x1fff) << 6  # picture_height
  if param:
    assert param.model_id < 16, 'model_id must be between 0 to 15'
    header += (param.model_id & 0x0f) << 2

  with open(output_fname, 'wb') as f:
    f.write(struct.pack('>I',header))
    f.write(lic_bitstream) 

# check if the bitstream file is image
# return header if the bitstream for image compression
# other wise return None
def get_image_bitstream_header(fname):
  
  with open(fname, 'rb') as f:
    data = f.read(4)
  if data == b'\x00\x00\x00\x01':
    return None # video bitstream

  header = SimpleNamespace()
  header_bin = struct.unpack('>I', data)[0]
  header.picture_width = header_bin >> 19
  header.picture_height = (header_bin >> 6) & 0x1fff
  header.model_id = (header_bin >> 2) & 0x0f
  return header

# constant definition for the bitstream
NALU_DELIMITER = b'\x00\x00\x01'
NALU_TYPE_NN_IRAP = 11

# check if the bitstream in intra fallback mode
# in intra fallback mode, the bitstream is a normal VVC bitstream. No NN IRAP NALU present in the bitstream.
def is_intra_fallback_mode(fname):
  with open(fname, 'rb') as f:
    data = f.read()
  
  # go through NAL units
  pos = 0
  num_nalu = 0
  nalu_start = -1
  while pos < len(data):
    if data[pos:pos+3] == NALU_DELIMITER: 
      if nalu_start != -1:
        nalu = data[nalu_start:pos]
        nul_unit_type = (nalu[1] >> 3) & 0x1f
        if nul_unit_type == NALU_TYPE_NN_IRAP: return False
        num_nalu += 1
      pos += 3
      nalu_start = pos
    else:
      pos += 1
  
  # the last NAL unit 
  nalu = data[nalu_start:pos]
  nul_unit_type = (nalu[1] >> 3) & 0x1f
  if nul_unit_type == NALU_TYPE_NN_IRAP: return False

  return True




# parser 

# parameters used internally has different name than the spec
# name mapping is performed here
_params_mapping = {
  'model_id' : 'intra_model_id',
  'IHA_flag' : 'intra_filter_flag',
  'IHA_patch' : 'intra_filter_patch_wise_flag',
  'IMA_model_id' : 'nnpfa_id',
}
_params_mapping_inverse = dict([reversed(x) for x in _params_mapping.items()])

# get video parameters
# param file is a text file in format
# key value
def get_params(fname):
  with open(fname, 'r') as f:
    data = csv.reader(f, delimiter=' ', skipinitialspace=True)                  
    data = [[_params_mapping_inverse[k1], int(v1)] for k1,v1 in data]
    params = SimpleNamespace(**dict(data))
  return params

# write params into a parameter file
# append: append to existing file
def write_params(fname, params, append=False):
  open_mode = 'w+' if append else 'w'
  with open(fname, open_mode) as f:
    writer = csv.writer(f, delimiter=' ')
    for key,value in params.__dict__.items():
      writer.writerow([_params_mapping[key], value])


# ZJU_BIT_STRUCT
# write restoration data into a restoration data file
def write_restoration_data(fname, item):
    # write parameters for frameMixxer.
    with open(fname, 'wb') as f:
      f.write(get_parameters_bitstream(item))
## end ZJU_BIT_STRUCT

def frame_iterator(fname):
  with open(fname, 'rb') as f:
    data = f.read()
    for idx in range(len(data)):
      frame_info = SimpleNamespace()
      frame_info.type = 'intra'
      yield frame_info
      #yield data[idx:idx+1]



_component_name_id_mapping = {
 'ROI':0, 
 'SpatialResample':1, 
 'TemporalResample':2, 
 'PostFilter':3,
 'BitDepthTruncation': 4,
 'InnerCodec':5
}
_id_component_name_mapping = dict([reversed(x) for x in _component_name_id_mapping.items()])

def get_parameters_bitstream(item):
  bs = bytearray()
  for component_name, param_data in item.parameters.items():
    bs.append(_component_name_id_mapping[component_name])
    bs += struct.pack('H', len(param_data)) # max length 64K
    bs += param_data
  bs.append(255) # end
  return bs

def parse_parameters_bitstream(bs):
  parameters = {}
  param_id = bs.read(1)[0]
  while param_id != 255:
    component_name = _id_component_name_mapping[param_id]
    param_length = struct.unpack_from('H', bs.read(2), 0)[0]
    param_data = bs.read(param_length)
    parameters[component_name] = param_data
    param_id = bs.read(1)[0]
  return parameters
  

def gen_vcm_bitstream(inner_bitstream_fname, item, out_bitstream_fname):
  '''
  Generate VCM bitstream. This is a temporary solution. The parameters required by post-inner 
  codec are added to the beginning of the inner bitsream
  '''
  with open(out_bitstream_fname, 'wb') as f:
    f.write(b'VCM') #identifier
    f.write(get_parameters_bitstream(item))
    with open(inner_bitstream_fname, 'rb') as inner_bs_f:
      inner_bs = inner_bs_f.read()
      f.write(inner_bs)

def parse_vcm_bitstream(vcm_bitstream_fname, inner_bitstream_fname, item):
  '''
  Parse VCM bitstream, getting parameters for post-inner codec components, and 
  set parameters in item

  Args:
    vcm_bitstream_fname
    innter_bitstream_fname
    item
  '''
  with open(vcm_bitstream_fname, 'rb') as f:
    data = f.read(3) # VCM header
    assert data==b'VCM', f'{vcm_bitstream_fname} is not a VCM bitstream file'
    item.parameters = parse_parameters_bitstream(f)
    data = f.read() # read rest of the data
    with open(inner_bitstream_fname, 'wb') as f2:
      f2.write(data)


