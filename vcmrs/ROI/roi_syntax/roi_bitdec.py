# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import bitstring as bs
from vcmrs.ROI.roi_utils import roi_utils
from vcmrs.ROI.roi_syntax import roi_consts

VERBOSE = False

def get_bits(stream, num_bits, default = 0, log = None, debug_text = None):
  if num_bits>0:
    ret = stream.read(num_bits).uint
    if VERBOSE and not log is None: log(debug_text+" read:"+str(ret))
    return ret
  if VERBOSE and not log is None: log(debug_text+" read default:"+str(default))
  return default

def get_signed_bits(stream, num_bits, default = 0, log = None, debug_text = None):
  if num_bits>0:
    ret = stream.read(num_bits).uint
    sig = stream.read(1).uint
    if sig>0:
      if VERBOSE and not log is None: log(debug_text+" read:"+str(-ret))      
      return -ret
    if VERBOSE and not log is None: log(debug_text+" read:"+str(ret))
    return ret
  if VERBOSE and not log is None: log(debug_text+" read default:"+str(default))
  return default
  
  
def decode_syntax_unit_retargeting_data(stream, log):
   
  bits_rtg_image_size = get_bits(stream, 5, log = log, debug_text = "bits_rtg_image_size")

  rtg_image_size_x = get_bits(stream, bits_rtg_image_size, log = log, debug_text = "rtg_image_size_x") 
  rtg_image_size_y = get_bits(stream, bits_rtg_image_size, log = log, debug_text = "rtg_image_size_y")

  if get_bits(stream, 1, log = log, debug_text = "flag_rtg_image_size_difference")==1:  
    bits_rtg_to_org_difference = get_bits(stream, 5, log = log, debug_text = "bits_rtg_to_org_difference")      
    rtg_to_org_difference_x = get_bits(stream, bits_rtg_to_org_difference, log = log, debug_text = "rtg_to_org_difference_x") 
    rtg_to_org_difference_y = get_bits(stream, bits_rtg_to_org_difference, log = log, debug_text = "rtg_to_org_difference_y") 
    org_image_size_x = rtg_image_size_x + rtg_to_org_difference_x
    org_image_size_y = rtg_image_size_y + rtg_to_org_difference_y
  else:
    org_image_size_x = rtg_image_size_x
    org_image_size_y = rtg_image_size_y
  
  rois = []
  if get_bits(stream, 1, log = log, debug_text = "flag_rtg_rois")==0:  
    rois.append( ( 0, (0, 0, org_image_size_x, org_image_size_y) ) )
    return rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, rois
  
  bits_roi_pos = roi_utils.get_required_num_of_bits_for_value( max(org_image_size_x, org_image_size_y) )  
  
  bits_scale_factor = roi_utils.get_required_num_of_bits_for_value(roi_consts.MAXIMAL_SCALE_FACTOR)  # 1 currently    # 4 previously
        
  bits_roi_size = get_bits(stream, 5, log = log, debug_text = "bits_roi_size")
  
  bits_num_rois = get_bits(stream, 4, log = log, debug_text = "bits_num_rois")
  
  num_rois = get_bits(stream, bits_num_rois, log = log, debug_text = "num_rois")
        
  current_roi_scale_factor = 0
  for r in range(num_rois):    
    if current_roi_scale_factor != roi_consts.MAXIMAL_SCALE_FACTOR:
      if get_bits(stream, 1, log = log, debug_text = "flag_roi_scale_factor")==1:        
        current_roi_scale_factor = get_bits(stream, bits_scale_factor, log = log, debug_text = "current_roi_scale_factor")    

    roi_pos_x  = get_bits(stream, bits_roi_pos, log = log, debug_text = "roi_pos_x")
    roi_pos_y  = get_bits(stream, bits_roi_pos, log = log, debug_text = "roi_pos_y")
    roi_size_x = get_bits(stream, bits_roi_size, log = log, debug_text ="roi_size_x") 
    roi_size_y = get_bits(stream, bits_roi_size, log = log, debug_text ="roi_size_y") 
    
    rois.append( ( current_roi_scale_factor, (roi_pos_x, roi_pos_y, roi_pos_x+roi_size_x, roi_pos_y+roi_size_y) ) )

  return rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, rois
  
  
  
def decode_roi_params(bytes_data, log = None):
      
  stream = bs.ConstBitStream(bytes=bytes_data)  

  # The data below is not essentially needed in standard
  # as "roi_update_period" should result from transmission of
  # adequate syntax element placeholders (NALU, SEI)
  # In VCMRS it is currently impossible to do so.
  # Therefore, data for all Intra Periods (Random Access points)
  # is transmitted sequentially, altogether in a single package.
    
  bits_roi_update_period = get_bits(stream, 5, log = log, debug_text = "bits_roi_update_period")
  roi_update_period = get_bits(stream, bits_roi_update_period, log = log, debug_text = "roi_update_period-1") + 1
 
  bits_roi_gops_num = get_bits(stream, 5, log = log, debug_text = "bits_roi_gops_num")
  roi_gops_num = get_bits(stream, bits_roi_gops_num, log = log, debug_text = "roi_gops_num") 
  
  # This data below is required in the standard:
  # ROI data is transmitted for each GOP/Intra period 
  # (Random Access point) indepedently
    
  rtg_image_size_x = None
  rtg_image_size_y = None
  org_image_size_x = None
  org_image_size_y = None  
  gops_rois = []  
  for gop_idx in range(roi_gops_num):
      
    rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, gop_rois = decode_syntax_unit_retargeting_data(stream, log)
    gops_rois.append(gop_rois)
    
  return roi_update_period, rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, gops_rois
    
