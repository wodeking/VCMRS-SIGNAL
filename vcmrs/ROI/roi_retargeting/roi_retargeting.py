# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import vcmrs
from numba import jit
from collections import deque
import torch
import sys
import re
import ast
import base64
import json
import subprocess
import shutil
import cv2
import numpy as np
from vcmrs.Utils import utils
from vcmrs.ROI.roi_utils import roi_utils
from vcmrs.ROI.roi_syntax import roi_consts

def find_ROIs_simple(objects_for_frames, feather, width, height, start_idx, end_idx):       
  outer_bbox = None

  for j in range(start_idx, end_idx+1):
    for i in range(len(objects_for_frames[j])):
      bbox = objects_for_frames[j][i]
      ebbox = roi_utils.rect_extend_limited(bbox, feather, width, height)
      #x1, y1, x2, y2 = roi_utils.rect_extend_limited(bbox, feather, width, height)
    
      if outer_bbox is None:
        outer_bbox = ebbox
        continue
      outer_bbox = ( min( ebbox[0], outer_bbox[0]), min( ebbox[1], outer_bbox[1]), max( ebbox[2], outer_bbox[2]), max( ebbox[3], outer_bbox[3]) )
  
  rois = []
  if outer_bbox is not None:
    rois.append( [ 0, outer_bbox ] ) # just one RoI so far
  return rois
  
  
  
def find_ROIs(objects_for_frames, feather, desired_max_obj_size, width, height, start_idx, end_idx, visualize_prefix = None):
  
  objects_for_all_frames = []
  for f in range(start_idx, end_idx+1):
    
    objects_for_frame_f = []
    for obj in objects_for_frames[f]:
      objects_for_frame_f.append( [obj[0], obj[1], obj[2], obj[3], 0] )
    
    optimized_rois = roi_utils.optimize_rois(objects_for_frame_f)
    

    for roi in optimized_rois:
      sx = roi[2]-roi[0]
      sy = roi[3]-roi[1]
      size = max(sx,sy)
      
      desired_size = min(size, desired_max_obj_size)
      desired_scale = desired_size / size
      
      desired_scale_factor = 0
      for scale_factor in range(roi_consts.MAXIMAL_SCALE_FACTOR):
        nom, den = roi_consts.retargeting_downscale_factors[scale_factor]
        scale = nom/den
        if scale>=desired_scale: desired_scale_factor = scale_factor
        
      roi[4] = desired_scale_factor

    if visualize_prefix is not None:
      roi_utils.visualize_rois(width, height, optimized_rois, f"{visualize_prefix}Objects_Frame{f}_Num{len(optimized_rois)}.png")

    objects_for_all_frames += optimized_rois
  
  if visualize_prefix is not None:
    roi_utils.visualize_rois(width, height, objects_for_all_frames, f"{visualize_prefix}Objects_All_Num{len(objects_for_all_frames)}.png")  
  
  optimized_objects_for_all_frames = roi_utils.optimize_rois(objects_for_all_frames)
  

  if visualize_prefix is not None:
    roi_utils.visualize_rois(width, height, optimized_objects_for_all_frames, f"{visualize_prefix}Objects_Opt_Num{len(optimized_objects_for_all_frames)}.png")
    
  rois = []
  for roi in optimized_objects_for_all_frames:
    scale = roi[4]
    roi = roi_utils.rect_extend_limited(roi, feather, width, height) # limit only
    rois.append( [ scale, roi ] ) 
    
  return rois

  
def find_ROIs_closing(objects_for_frames, feather, width, height, start_idx, end_idx, visualize_prefix = None):

  #closing = 160
  #closing = 80
  closing = 40
  #closing = 256
  
  img = np.zeros( (height, width), dtype=np.uint8)
    
  mask_closing = roi_utils.get_mask_with_feather(objects_for_frames, width, height, feather+closing, 128, start_idx, end_idx)
  #cv2.imwrite("debug1_closing.png",mask_closing) 
  
  #obj.min_x = obj.box[0]
  #obj.min_y = obj.box[1]
  #obj.size_x = obj.box[2]
  #obj.size_y = obj.box[3]

  
  mask_closing = roi_utils.erode_image(mask_closing, closing*2+1)
  #cv2.imwrite("debug2_closing.png",mask_closing) 

  num_labels, labels = cv2.connectedComponents(mask_closing)
  #cv2.imwrite("debug3_labels.png",labels) 
  
  
  rois = []
  for lab in range(1, num_labels):
    #mask = np.zeros( (height, width), dtype=np.uint8)
    #mask[labels == label] = 255
    mask = (labels==lab).astype(np.uint8)
    #cv2.imwrite(f"debug4_mask{lab}.png",mask.astype(np.uint8)*255) 
    rect = cv2.boundingRect(mask)
    roi = ( rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3] )
    
    roi = roi_utils.rect_extend_limited(roi, 0, width, height) # limit only
    rois.append( [ 0, roi ] ) 
  
  return rois
  

def find_ROIs_closing_margin(objects_for_frames, feather, width, height, start_idx, end_idx, visualize_prefix = None):

  #closing = 160
  #closing = 80
  closing = 40
  #closing = 256
  
  img = np.zeros( (height, width), dtype=np.uint8)
    
  margin = feather+closing
  #mask_closing = roi_utils.get_mask_with_feather(objects_for_frames, width, height, feather+closing, 128, start_idx, end_idx)
  mask_closing = roi_utils.get_mask_with_feather(objects_for_frames, width+margin*2, height+margin*2, feather+closing, 128, start_idx, end_idx, offset_x=margin, offset_y=margin)
  #cv2.imwrite("debug1_closing.png",mask_closing) 
  
  #obj.min_x = obj.box[0]
  #obj.min_y = obj.box[1]
  #obj.size_x = obj.box[2]
  #obj.size_y = obj.box[3]

  
  mask_closing = roi_utils.erode_image(mask_closing, closing*2+1)
  #cv2.imwrite("debug2_closing.png",mask_closing) 

  mask_closing = mask_closing[margin:margin+height, margin:margin+width]
  #cv2.imwrite("debug3_closing.png",mask_closing) 

  num_labels, labels = cv2.connectedComponents(mask_closing)
  #cv2.imwrite("debug4_labels.png",labels) 
  
  rois = []
  for lab in range(1, num_labels):
    #mask = np.zeros( (height, width), dtype=np.uint8)
    #mask[labels == label] = 255
    mask = (labels==lab).astype(np.uint8)
    #cv2.imwrite(f"debug5_mask{lab}.png",mask.astype(np.uint8)*255) 
    rect = cv2.boundingRect(mask)
    roi = ( rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3] )
    
    roi = roi_utils.rect_extend_limited(roi, 0, width, height) # limit only
    rois.append( [ 0, roi ] ) 
  
  return rois
  
def _generate_grid_coords(rois, org_size_x, org_size_y):
  org_coords_x = [0, org_size_x]
  org_coords_y = [0, org_size_y]
       
  for roi in rois:                
    scale_factor = roi[0]
    rect = roi[1]
    
    start_x = rect[0]
    end_x   = rect[2]
    
    start_y = rect[1]
    end_y   = rect[3]
    
    if not start_x in org_coords_x:
      org_coords_x.append(start_x)      
    if not end_x in org_coords_x:
      org_coords_x.append(end_x)      
    
    if not start_y in org_coords_y:
      org_coords_y.append(start_y)      
    if not end_y in org_coords_y:
      org_coords_y.append(end_y)           
      
  org_coords_x.sort()
  org_coords_y.sort()
  return org_coords_x, org_coords_y


def _generate_scale_factors_grid(rois, org_coords_x, org_coords_y):
  bsize_x = len(org_coords_x)
  bsize_y = len(org_coords_y)
  
  max_scale_factor = roi_consts.MAXIMAL_SCALE_FACTOR
  scale_factors_grid = np.full( (bsize_y-1, bsize_x-1), dtype=np.uint8, fill_value=max_scale_factor)
      
  for roi in rois:                
    scale_factor = roi[0]
    rect = roi[1]
    
    start_x = org_coords_x.index(rect[0])
    end_x   = org_coords_x.index(rect[2])
    
    start_y = org_coords_y.index(rect[1])
    end_y   = org_coords_y.index(rect[3])
    scale_factors_grid[start_y:end_y, start_x:end_x] = np.minimum( scale_factors_grid[start_y:end_y, start_x:end_x], scale_factor)
    #for by in range(start_y, end_y):
    #  for bx in range(start_x, end_x):
    #    scale_factors_grid[by,bx] = min(scale_factors_grid[by,bx], scale_factor)

  return scale_factors_grid

def _retarget_coords(org_coords, scale_factors):
  bsize = len(org_coords)
  new_size = 0
  new_coords = [0]
  for b in range(bsize-1):    
    
    scale_factor = scale_factors[b]
    scale_factor_nom, scale_factor_den = roi_consts.retargeting_downscale_factors[scale_factor]    
    org_roi_size = (org_coords[b+1]-org_coords[b])
    nom_org_roi_size = org_roi_size * scale_factor_nom
    new_roi_size  = nom_org_roi_size // scale_factor_den
    new_size += new_roi_size
    new_coords.append(new_size)
  return new_coords, new_size

def _total_retarget_coords(org_coords, scale_factors, align_size, expected_size):
  new_coords, new_size = _retarget_coords(org_coords, scale_factors)
  
  if not align_size is None:
    if not expected_size is None:
      vcmrs.log("Both align_size and expected_size cannot be not None")
      exit()
  
  if not align_size is None:
    if align_size>1:
      new_size_corrected = ((new_size+align_size-1)//align_size)*align_size
      if new_size_corrected != new_size:
        expected_size = new_size_corrected
      
  if not expected_size is None:
    #vcmrs.log(expected_size)
    if new_size>0:
      for i in range(len(new_coords)):
        new_coords[i] = new_coords[i]*(expected_size)//(new_size)
    else:
      new_coords[-1] = expected_size

    new_size = new_coords[-1]
    if new_size!=expected_size:
      vcmrs.log(f"RetargetingROI: SIZE ERROR:  {new_size} vs {expected_size}")
      exit()
  
  return new_coords, new_size


def generate_coords(org_size_x, org_size_y, rois, align_size_x, align_size_y, expected_rtg_size_x, expected_rtg_size_y):

  org_coords_x, org_coords_y = _generate_grid_coords(rois, org_size_x, org_size_y)
  
  scale_factors_grid = _generate_scale_factors_grid(rois, org_coords_x, org_coords_y)
     
  bsize_x = len(org_coords_x)
  bsize_y = len(org_coords_y)
  
  x_wise_scale_factors = []
  for bx in range(bsize_x-1):
    minimal_scale_factor = min(scale_factors_grid[:,bx])
    x_wise_scale_factors.append(minimal_scale_factor)
      
  y_wise_scale_factors = []      
  for by in range(bsize_y-1):
    minimal_scale_factor = min(scale_factors_grid[by,:])
    y_wise_scale_factors.append(minimal_scale_factor)
  
  rtg_coords_x,rtg_size_x = _total_retarget_coords(org_coords_x, x_wise_scale_factors, align_size_x, expected_rtg_size_x)    
  rtg_coords_y,rtg_size_y = _total_retarget_coords(org_coords_y, y_wise_scale_factors, align_size_y, expected_rtg_size_y)
  
  return org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y

def retarget_image(inp_image, inp_coords_x, inp_coords_y, out_coords_x, out_coords_y):  
  bsize_x = len(inp_coords_x)
  bsize_y = len(inp_coords_y)
  
  inp_image_size_x = inp_coords_x[-1]
  inp_image_size_y = inp_coords_y[-1]
      
  out_image_size_x = out_coords_x[-1]
  out_image_size_y = out_coords_y[-1]
    
  v_sum = np.array([0,0,0], dtype = np.uint32)
  v_num = 0  
  
  out_image = np.zeros( (out_image_size_y, out_image_size_x, 3), np.uint8) 
  for by in range(bsize_y-1):        
    for bx in range(bsize_x-1):  
      inp_start_x = inp_coords_x[bx]
      inp_start_y = inp_coords_y[by]
      inp_end_x   = inp_coords_x[bx+1]
      inp_end_y   = inp_coords_y[by+1]
      inp_size_x = inp_end_x - inp_start_x
      inp_size_y = inp_end_y - inp_start_y
        
      out_start_x = out_coords_x[bx]
      out_start_y = out_coords_y[by]
      out_end_x   = out_coords_x[bx+1]
      out_end_y   = out_coords_y[by+1]
      out_size_x = out_end_x - out_start_x
      out_size_y = out_end_y - out_start_y
      
      if (out_size_x==0) or (out_size_y==0): continue
      
      if (inp_size_x==0) or (inp_size_y==0): 
        
        inp_start_x1 = max(inp_start_x-1,0)
        inp_start_y1 = max(inp_start_y-1,0)
        
        inp_end_x1 = min(inp_end_x+1,inp_image_size_x)
        inp_end_y1 = min(inp_end_y+1,inp_image_size_y)
        
        num = (inp_end_x1-inp_start_x1) * (inp_end_y1-inp_start_y1)
        if num>0:          
          v_sum += np.sum( np.sum(inp_image[ inp_start_y1:inp_end_y1 , inp_start_x1:inp_end_x1 ], axis=0), axis=0 )
          v_num += num
        continue

      if (inp_size_x==out_size_x) and (out_size_y==inp_size_y):
        try:
          out_image[ out_start_y:out_end_y , out_start_x:out_end_x ] = inp_image[ inp_start_y:inp_end_y , inp_start_x:inp_end_x ]
        except:
          vcmrs.log("shape:"+str(out_image.shape))
          vcmrs.log(f"osy{out_start_y} oey{out_end_y} osx{out_start_x} oex{out_end_x} isy{inp_start_y} iey{inp_end_y} isx{inp_start_x} iex{inp_end_x}" )
          assert 0
        continue
  
      M = cv2.getRotationMatrix2D( (0,0) ,0, 0 ) 
      
      M[0,0] = (inp_size_x-1)/(out_size_x-1) if out_size_x>1 else 0
      M[0,1] = 0
      M[0,2] = inp_start_x
      
      M[1,0] = 0
      M[1,1] = (inp_size_y-1)/(out_size_y-1) if out_size_y>1 else 0
      M[1,2] = inp_start_y
    
      #M = ( ( (inp_size_x)/(out_size_x) ,                      0  , inp_start_x),
      #      ( 0                         ,  inp_size_x/out_size_x  , inp_start_y) )
  
      #dst(x,y) = src( M11*x + M12*y + M13,   M21*x + M22*y + M23)
     
      block_resized = cv2.warpAffine(inp_image,M,(out_size_x,out_size_y),flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP )
      
      out_image[ out_start_y:out_end_y , out_start_x:out_end_x ] = block_resized
  
  if v_num>0:
    vset = v_sum / v_num
  else:
    vset = 127
        
  for by in range(bsize_y-1):        
    for bx in range(bsize_x-1):  
      inp_start_x = inp_coords_x[bx]
      inp_start_y = inp_coords_y[by]
      inp_end_x   = inp_coords_x[bx+1]
      inp_end_y   = inp_coords_y[by+1]
      inp_size_x = inp_end_x - inp_start_x
      inp_size_y = inp_end_y - inp_start_y

      out_start_x = out_coords_x[bx]
      out_start_y = out_coords_y[by]
      out_end_x   = out_coords_x[bx+1]
      out_end_y   = out_coords_y[by+1]
      out_size_x = out_end_x - out_start_x
      out_size_y = out_end_y - out_start_y
      
      if (out_size_x==0) or (out_size_y==0): continue
      
      if (inp_size_x==0) or (inp_size_y==0): 
        out_image[ out_start_y:out_end_y , out_start_x:out_end_x ] = vset
        
  return out_image        


