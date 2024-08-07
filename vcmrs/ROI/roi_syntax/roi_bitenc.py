# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import numpy as np
from bitstring import BitArray
from bitstring import Bits
from vcmrs.ROI.roi_utils import roi_utils
from vcmrs.ROI.roi_syntax import roi_consts

VERBOSE = False
      
def sgu(val): # sign 
  return 0 if val>=0 else 1 
  

def put_bits(result_stream, num_of_bits, value, log = None, debug_text = None):
  if num_of_bits>0:   
    result_stream.append(Bits(uint=value        ,length=num_of_bits ))
    if VERBOSE and log is not log: log(debug_text+" encoded:"+str(value))
  else:
    if VERBOSE and log is not log: log(debug_text+" encoded 0 bits")
     
def put_signed_bits(result_stream, num_of_bits, value, log = None, debug_text = None):
  if num_of_bits>0:   
    result_stream.append(Bits(uint=abs(value)      ,length=num_of_bits ))
    result_stream.append(Bits(uint=sgu(value)      ,length=1 ))
    if VERBOSE and log is not None: log(debug_text+" encoded: "+str(value))
  else:
    if VERBOSE and log is not None: log(debug_text+" encoded 0 bits")
   
   
def encode_syntax_unit_retargeting_data(result, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, rois, log):

  bits_rtg_image_size = roi_utils.get_required_num_of_bits_for_value( max(rtg_image_size_x, rtg_image_size_y) )
    
  put_bits(result, 5, bits_rtg_image_size, log = log, debug_text = "bits_rtg_image_size")  # max img size   = 2**31-1 = 2147483647
    
  put_bits(result, bits_rtg_image_size, rtg_image_size_x, log = log, debug_text = "rtg_image_size_x")  
  put_bits(result, bits_rtg_image_size, rtg_image_size_y, log = log, debug_text = "rtg_image_size_y")
    
  
  if (org_image_size_x!=rtg_image_size_x) or (org_image_size_y!=rtg_image_size_y):
  
    rtg_to_org_difference_x = org_image_size_x - rtg_image_size_x
    rtg_to_org_difference_y = org_image_size_y - rtg_image_size_y
  
    put_bits(result, num_of_bits=1, value=1, log = log, debug_text = "flag_rtg_image_size_difference")
  
    bits_rtg_to_org_difference = roi_utils.get_required_num_of_bits_for_value( max(rtg_to_org_difference_x, rtg_to_org_difference_y) )  
  
    put_bits(result, 5, bits_rtg_to_org_difference, log = log, debug_text = "bits_rtg_to_org_difference")  # max size difference   = 2**31-1 = 2147483647
    
    put_bits(result, bits_rtg_to_org_difference, rtg_to_org_difference_x, log = log, debug_text = "rtg_to_org_difference_x")  
    put_bits(result, bits_rtg_to_org_difference, rtg_to_org_difference_y, log = log, debug_text = "rtg_to_org_difference_y")
  else:
    put_bits(result, num_of_bits=1, value=0, log = log, debug_text = "flag_rtg_image_size_difference")
      
  num_rois = len(rois)
  
  if num_rois==1:
    scale_factor, bbox = rois[0]
    if (scale_factor==0) and (bbox[0]==0) and (bbox[1]==0) and (bbox[2]==org_image_size_x) and (bbox[3]==org_image_size_y):      
      put_bits(result, num_of_bits=1, value=0, log = log, debug_text = "flag_rtg_rois")
      return
      
  put_bits(result, num_of_bits=1, value=1, log = log, debug_text = "flag_rtg_rois")

  bits_roi_pos = roi_utils.get_required_num_of_bits_for_value( max(org_image_size_x, org_image_size_y) )  

  max_sx = 0
  max_sy = 0
  count = 0
  for scale_factor, bbox in rois:

    sx = bbox[2]-bbox[0]
    sy = bbox[3]-bbox[1]
    max_sx = max(max_sx, sx)
    max_sy = max(max_sy, sy)
    count += 1
  
  bits_roi_size = roi_utils.get_required_num_of_bits_for_value( max(max_sx, max_sy) )  

  put_bits(result, 5, bits_roi_size, log = log, debug_text = "bits_roi_size")  # max roi size   = 2**31-1 = 2147483647
  
  bits_num_rois = roi_utils.get_required_num_of_bits_for_value(num_rois)

  put_bits(result, 4, bits_num_rois, log = log, debug_text = "bits_num_rois")  # max num rois   = 2**15-1 = 32767  
  put_bits(result, bits_num_rois, num_rois, log = log, debug_text = "num_rois")
  
  bits_scale_factor = roi_utils.get_required_num_of_bits_for_value(roi_consts.MAXIMAL_SCALE_FACTOR)  # 1 currently    # 4 previously

  current_scale_factor = 0
  for scale_factor, bbox in rois:    
    sx = bbox[2]-bbox[0]
    sy = bbox[3]-bbox[1]    
    if current_scale_factor != roi_consts.MAXIMAL_SCALE_FACTOR:
      if current_scale_factor != scale_factor: #different scale_factor, indicate '1' flag and scale_factor
        put_bits(result, num_of_bits=1, value=1, log = log, debug_text = "flag_roi_scale_factor")        
        put_bits(result, bits_scale_factor, scale_factor, log = log, debug_text = "roi_scale_factor")
      else:   #scale_factor as previous, write '0'
        put_bits(result, num_of_bits=1, value=0, log = log, debug_text = "flag_roi_scale_factor")        
    
    #write cooardinates x,y, dim hor,ver
    put_bits(result, bits_roi_pos  ,bbox[0], log = log, debug_text = "roi_pos_x")
    put_bits(result, bits_roi_pos  ,bbox[1], log = log, debug_text = "roi_pos_y")
    put_bits(result, bits_roi_size ,sx     , log = log, debug_text = "roi_size_x")
    put_bits(result, bits_roi_size ,sy     , log = log, debug_text = "roi_size_y")
    
    current_scale_factor = scale_factor
  
def encode_gops_rois_params(roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, gops_rois, log):

    result = BitArray()
    
    # The data below is not essentially needed in standard
    # as "roi_update_period" should result from transmission of
    # adequate syntax element placeholders (NALU, SEI)
    # In VCMRS it is currently impossible to do so.
    # Therefore, data for all Intra Periods (Random Access points)
    # is transmitted sequentially, altogether in a single package.
    
    bits_roi_update_period = roi_utils.get_required_num_of_bits_for_value(roi_update_period-1)
    put_bits(result, 5, bits_roi_update_period, log = log, debug_text = "bits_roi_update_period")
    put_bits(result, bits_roi_update_period, roi_update_period-1, log = log, debug_text = "roi_update_period-1")
    
    roi_gops_num = 0
    if not gops_rois is None:
      roi_gops_num = len(gops_rois)
    
    bits_roi_gops_num = roi_utils.get_required_num_of_bits_for_value(roi_gops_num)
      
    put_bits(result, 5, bits_roi_gops_num, log = log, debug_text = "bits_roi_gops_num")  # max num of gops = 2**31-1 = 2147483647
    put_bits(result, bits_roi_gops_num, roi_gops_num, log = log, debug_text = "roi_gops_num")    
    
    # This data below is required in the standard:
    # ROI data is transmitted for each GOP/Intra period 
    # (Random Access point) indepedently
    
    for gop_idx in range(roi_gops_num):
      
      if gops_rois is not None:
        rois = gops_rois[gop_idx]
        encode_syntax_unit_retargeting_data(result, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, rois, log)    
  
    result_bytes = result.tobytes()

    return result_bytes
