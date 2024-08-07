# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import vcmrs
import sys
import cv2
import numpy as np
import ast
#from vcmrs.Utils import utils

# rect is defined as (x1, y1, x2, y2)  therefore usage for adressing image is as follows: [y1:y2, x1:x2]

def rect_extend_limited(rect, ext, width, height):
  minx = max(0,rect[0]-ext)
  miny = max(0,rect[1]-ext)

  maxx = min(rect[2]+ext, width)
  maxy = min(rect[3]+ext, height)

  return [minx, miny, maxx, maxy]

def rect_extend(rect, ext):
  return [ rect[0]-ext, rect[1]-ext, rect[2]+ext, rect[3]+ext ]

def rect_offset(rect, offs_x, offs_y):
  return [ rect[0]+offs_x, rect[1]+offs_y, rect[2]+offs_x, rect[3]+offs_y ] 

def rect_limit(rect, width, height):
  return [ max(0,rect[0]), max(0,rect[1]), min(rect[2], width), min(rect[3], height) ] 

# fc as a fraction of sampling frequency
def get_kaiser_sinc_kernel(fc, N = 25):
  assert fc > 0.0 and fc <= 0.5
  assert N % 2 == 1
  kernel = np.sinc(2 * fc * (np.arange(N) - (N - 1) / 2) )
  beta = 1.143  # Kaiser window beta.
  window = np.kaiser( N, beta)
  kernel *= window
  kernel /= np.sum(kernel)
  return kernel

def get_lanczos_kernel(fc, N = 25):
  assert fc > 0.0 and fc <= 0.5
  assert N % 2 == 1
  r = N // 2
  kernel = np.sinc(2 * fc * (np.arange(-r, r+1)) )
  window = np.sinc( np.arange(-r, r+1)/(r+1) )
  kernel *= window
  kernel /= np.sum(kernel)
  return kernel

def convert_coeffs_to_int_div(coeffs_f, coeff_scale, coeff_type):
  # in general:  return np.around(coeffs_f * coeff_scale).astype(np.int32)
  
  coeffs_f_sum = np.sum(coeffs_f)
  coeffs_i = np.around(coeffs_f * coeff_scale).astype(coeff_type)
  coeffs_i_sum = np.sum(coeffs_i)
  
  div = coeffs_i_sum/coeffs_f_sum
  div_i = np.around(div).astype(coeff_type)
  
  return coeffs_i, div_i

def coord_border_replicate(coord, size): # aaaaaa|abcdefgh|hhhhhhh
  return 0 if coord<0 else size-1 if coord>=size else coord

def coord_border_reflect(coord, size): # fedcba|abcdefgh|hgfedcb
  size2 = size*2
  pos = coord % size2
  # 01234567 89ABCDEF
  # abcdefgh|hgfedcba
  return pos if pos<size else size2 - 1 - pos
  return res

def coord_border_reflect_101(coord, size): # gfedcb|abcdefgh|gfedcba   # BORDER_DEFAULT
  size2_2 = size*2-2
  pos = coord % size2_2
  # 01234567 89ABCD
  # abcdefgh|gfedcb
  return pos if pos<size else size2_2 - pos

def coord_border_wrap(coord, size): # cdefgh|abcdefgh|abcdefg
  return coord % size

def filter_get_min_max_output(h, min_value, max_value):
  res_min = 0.0
  res_max = 0.0
  for coeff in h:
    if coeff>0:
      res_max += max_value*coeff
      res_min += min_value*coeff
    else:
      res_max += min_value*coeff
      res_min += max_value*coeff
      
  return res_min, res_max

def filter_sep_2D(image, hx, hy, border, implementation, min_value, max_value, ret_dtype):

  # implementation can be:
  # - integer   - number of bits for integer coefficients representation, e.g. 23
  # - np.int64  - integer coefficients representation, the best possible to fit inside 64 bits during computations
  # - "opencv"  - numpy @ float implementation
  # - "float"   - numpy @ integer implementation
  
  if implementation == "opencv": # use OpenCV pipeline which is based on floating point operations which are not cross-platform-identical
    if border == 'replicate':    borderType = cv2.BORDER_REPLICATE 
    if border == 'reflect':      borderType = cv2.BORDER_REFLECT
    if border == 'reflect_101':  borderType = cv2.BORDER_REFLECT_101
    if border == 'wrap':         borderType = cv2.BORDER_WRAP

    result = cv2.sepFilter2D(image, -1, hx, hy, borderType=borderType)
    result = np.clip( result, min_value, max_value ).astype(ret_dtype)
    return result
    
  if border == 'replicate':    borderFunc = coord_border_replicate
  if border == 'reflect':      borderFunc = coord_border_reflect
  if border == 'reflect_101':  borderFunc = coord_border_reflect_101
  if border == 'wrap':         borderFunc = coord_border_wrap

  try:
    integer_coeffs_bits = int(implementation) # number of bits for integer implementation
  except:
    integer_coeffs_bits = None
  
  # numpy @ float implementation
  hx_rounding_offset = 0
  hy_rounding_offset = 0
  use_dtype = np.float32
  hxy_div = 1

  HX = len(hx)
  RX = HX // 2
  HY = len(hy)
  RY = HY // 2
  
  hx_min, hx_max = filter_get_min_max_output(hx, min_value, max_value)
  hy_min, hy_max = filter_get_min_max_output(hy, hx_min, hx_max)
  h_amax = max(hy_min, hy_max)
  bits = get_required_num_of_bits_for_value(np.ceil(h_amax))

  # numpy @ integer implementations:

  # target computational bit depth: np.int64
  if implementation == np.int64:
    assert bits<63
    integer_coeffs_bits = (63-bits)//2
    
  # expected coefficient bit-depth
  if integer_coeffs_bits is not None:
    coeff_scale = (1<<integer_coeffs_bits)
    
    bits_required = integer_coeffs_bits*2 + bits
    
    use_dtype = np.int16
    if (bits_required>15): use_dtype = np.int32
    if (bits_required>31): use_dtype = np.int64
    if (bits_required>63): use_dtype = np.float64

    coeff_scale = (1<<integer_coeffs_bits)
    hx, hx_div = convert_coeffs_to_int_div(hx, coeff_scale, use_dtype)
    hy, hy_div = convert_coeffs_to_int_div(hy, coeff_scale, use_dtype)

    hxy_div = hy_div * hy_div
    hxy_rounding = hxy_div//2

  is_mono = len(image.shape)==2

  if is_mono:
    size_y, size_x = image.shape
    comps = 1
    image = image.reshape( (size_y, size_x, 1) )
  else:
    size_y, size_x, comps = image.shape
  
  image_hx = np.zeros( (size_y       , size_x + HX-1, comps), dtype = np.uint8 )
  image_hy = np.zeros( (size_y + HY-1, size_x       , comps), dtype = use_dtype )
  image_o  = np.zeros( (size_y       , size_x       , comps), dtype = ret_dtype )
  
  image_hx[:, RX:size_x+RX, :] = image[:,:,:]

  for x in range(RX):
    image_hx[:,x,:] = image_hx[:,borderFunc(x-RX, size_x)+RX,:]
  for x in range(size_x+RX, size_x+HX-1):
    image_hx[:,x,:] = image_hx[:,borderFunc(x-RX, size_x)+RX,:]
  
  # filter horizontally
  for y in range(size_y): 
    for c in range(comps):
      con = np.convolve(image_hx[y,:,c], hx, mode="valid")
      image_hy[y+RY,:,c] = con

  for y in range(RY):
    image_hy[y,:,:] = image_hy[borderFunc(y-RY, size_y)+RY,:,:]
  for y in range(size_y+RY, size_y+HY-1):
    image_hy[y,:,:] = image_hy[borderFunc(y-RY, size_y)+RY,:,:]
  
  # filter vertically
  for x in range(size_x): 
    for c in range(comps):
      con = np.convolve(image_hy[:,x,c], hy, mode="valid")
      if hxy_div!=1:
        con = (con+hxy_rounding) // hxy_div
      image_o[:,x,c] = np.clip( con, min_value, max_value)

  if is_mono:
    return image_o.reshape( (size_y, size_x) )

  return image_o

def filter_image_lowpass(image, fcx, fcy, N, filter_method, border, implementation):  #'lanczos',  'replicate'
  assert filter_method in ['lanczos', 'kaiser']

  if filter_method == 'lanczos': get_1D_kernel = get_lanczos_kernel
  if filter_method == 'kaiser':  get_1D_kernel = get_kaiser_sinc_kernel
  
  if fcx > 0.5:   hx = np.array([1], dtype=float) ## upscaling case
  else:           hx = get_1D_kernel(fcx, N)
  
  if fcy > 0.5:   hy = np.array([1], dtype=float) ## upscaling case
  else:           hy = get_1D_kernel(fcy, N)
  
  
  return filter_sep_2D(image, hx, hy, border, implementation, 0, 255, np.uint8)
  
def erode_image(image, N):
  kernel = np.ones((N,N),np.uint8)
  return cv2.erode(image,kernel,iterations = 1)
  
  #kernel = np.ones((1,N),np.uint8)
  #image = cv2.erode(image,kernel,iterations = 1)
  #kernel = np.ones((N,1),np.uint8)
  #image = cv2.erode(image,kernel,iterations = 1)
  #return image

def blur_background_scaled(ker, any_object, org_image, fc, scale):
  org_width = org_image.shape[1] 
  org_height = org_image.shape[0] 


  width = int(org_width /scale)
  height = int(org_height /scale)
  dim = (width, height)
  
  resized_any_object = cv2.resize(any_object, dim, interpolation = cv2.INTER_LINEAR)
  resized_org_image  = cv2.resize(org_image,  dim, interpolation = cv2.INTER_LINEAR)
  
  resized_inpainted_image = cv2.inpaint(resized_org_image,resized_any_object,5,cv2.INPAINT_NS)
  
  resized_blurred_image = filter_image_lowpass(resized_inpainted_image, fc, fc, ker, 'lanczos', 'replicate', np.int64)
  
  blurred_image = cv2.resize(resized_blurred_image, (org_width, org_height), interpolation = cv2.INTER_LINEAR)

  return blurred_image

def mix_images(img0, mask1, slope_size_half, img1, out_img = None):

  if slope_size_half>0:
    t = np.arange(-slope_size_half, slope_size_half+1)/(slope_size_half+1)
    h = ( np.cos( t*np.pi )+1 ) / ( slope_size_half*2 + 2 )
    alpha1 = filter_sep_2D(mask1, h, h, 'reflect', np.int64, 0, 128, np.uint8)
    alpha0 = 128-alpha1
  else:
    alpha1 = mask1
    alpha0 = 128-mask1
  
  if out_img is None:
    out_img = np.zeros( (img0.shape[0], img0.shape[1], img0.shape[2]),dtype=np.uint8)
    
  for c in range(img0.shape[2]):
    p0 = np.multiply( img0[:,:,c], alpha0, dtype = np.uint16)
    p1 = np.multiply( img1[:,:,c], alpha1, dtype = np.uint16)
    out_img[:,:,c] = (p0 + p1)>>7

  return out_img    

def get_mask_with_feather(objects_for_frames, width, height, feather, value, start_idx, end_idx, mask = None, offset_x=0, offset_y=0):
  if mask is None:
    mask = np.zeros((height, width),dtype=np.uint8)  
  for j in range(start_idx, end_idx+1):        
    for i in range(len(objects_for_frames[j])):
      rect = objects_for_frames[j][i]
      rect = rect_extend(rect, feather)
      rect = rect_offset(rect, offset_x, offset_y)
      rect = rect_limit(rect, width, height)
      x1, y1, x2, y2 = rect
      mask[y1:y2, x1:x2] = value
  return mask
  
def add_mask_with_feather(objects_for_frames, width, height, feather, value, start_idx, end_idx, mask):
  for j in range(start_idx, end_idx+1):        
    for i in range(len(objects_for_frames[j])):
      rect = objects_for_frames[j][i]
      x1, y1, x2, y2 = rect_extend_limited(rect, feather, width, height)
    
      mask[y1:y2, x1:x2] += value
  
def load_descriptor_from_file(descriptor_file):
  with open(descriptor_file, 'r') as file:
    content = file.read()
    return ast.literal_eval(content)
      
def save_descriptor_to_file(descriptor_file, content):
  with open(descriptor_file, 'w') as file:
    file.write(str(content))
      
def get_required_num_of_bits_for_value(val):
  n = int(np.ceil(np.log2(val+1)))
  if (val>=(1<<n)): # required due to potential numerical inaccuracy of ceil function
    n += 1
  return n

def rect_area(r):
  if r is None:
    return 0
  return (r[2]-r[0])*(r[3]-r[1])

def rect_intersection(a, b):
  maxx = min(a[2], b[2])
  maxy = min(a[3], b[3])
    
  minx = max(a[0], b[0])
  miny = max(a[1], b[1])
    
  if (maxx<=minx):
    return None
    
  if (maxy<=miny):
    return None
      
  return [minx, miny, maxx, maxy]

def rect_bounding(a, b):
  maxx = max(a[2], b[2])
  maxy = max(a[3], b[3])
    
  minx = min(a[0], b[0])
  miny = min(a[1], b[1])
    
  if (maxx<=minx):
    return None
    
  if (maxy<=miny):
    return None
      
  return [minx, miny, maxx, maxy]

def rect_extend_simple(rect, ext):
  return [rect[0]-ext, rect[1]-ext, rect[2]+ext, rect[3]+ext]
  
def rect_shrink_simple(rect, ext):
  return [rect[0]+ext, rect[1]+ext, rect[2]-ext, rect[3]-ext]
        
ROI_OPTIMIZE_BOX_EXTENSION = 30
        
def optimize_rois(i_rois):
  o_rois = []
  #for i_scale, i_rect in i_rois:
  for i_rect in i_rois:
    i_scale =0
    i_s = i_rect[4]
    
    o_idx = 0
    while o_idx<len(o_rois):
      #otuple = o_rois[o_idx]
      #o_scale = otuple[0]
      #o_rect = otuple[1]
      o_rect = o_rois[o_idx]
      o_scale = 0
      o_s = o_rect[4]
      

      if (i_scale<o_scale): # potentially new one covers old one  
        ise_rect = rect_intersection(i_rect,o_rect)
        ise_area = rect_area(ise_rect) # area is lesser than of any of them
        
        o_area = rect_area(o_rect)
        IoS = ise_area / o_area
        if IoS > 0.99: # old is almost covered, so delete
          del o_rois[o_idx] # and continue
          continue
          
        o_idx += 1
        continue      
        
      if (o_scale<i_scale): # potentially old one covers new one 
        ise_rect = rect_intersection(i_rect,o_rect)
        ise_area = rect_area(ise_rect) # area is lesser than of any of them
        
        i_area = rect_area(i_rect)
        IoN = ise_area / i_area
        if IoN > 0.99: # new one is almost entirelt covered, so do not append
          i_rect = None  
          break # and do not go through others
        o_idx += 1        
        continue      
        
      if (o_scale!=i_scale):
        o_idx += 1
        continue
        
      i_rect_ext = rect_extend_simple(i_rect, ROI_OPTIMIZE_BOX_EXTENSION)
      o_rect_ext = rect_extend_simple(o_rect, ROI_OPTIMIZE_BOX_EXTENSION)
      
      ise_rect = rect_intersection(i_rect_ext,o_rect_ext)
      bou_rect = rect_bounding(i_rect_ext,o_rect_ext)
      if bou_rect is None:
        o_idx += 1
        continue
        
      ise_area = rect_area(ise_rect)
      bou_area = rect_area(bou_rect)
      
      IoB = ise_area/bou_area
      if IoB>0.85: # common part of old one with new one covers most of area of their bounding box, so lets replace old (one and new one also) with bounding boxem
        del o_rois[o_idx]
        bou_rect = rect_shrink_simple(bou_rect, ROI_OPTIMIZE_BOX_EXTENSION)
        i_rect = bou_rect + [min(i_s, o_s)] # we will add it later
        #break
        continue
      
      i_area = rect_area(i_rect_ext)
      o_area = rect_area(o_rect_ext)
      
      area_uni = i_area + o_area - ise_area # powierzchnia union = jeden lub drugi
      
      UoB = area_uni/bou_area   
      if UoB > 0.95: # area of union covers entirety of bounding boxa, so lets replace old one (and new one also) with bounding box
        del o_rois[o_idx]
        bou_rect = rect_shrink_simple(bou_rect, ROI_OPTIMIZE_BOX_EXTENSION)  # i_scale as well
        i_rect = bou_rect + [min(i_s, o_s)] # we will add it later
        #break
        continue
      o_idx += 1
      
    if not i_rect is None:
      #o_rois.append( (i_scale, i_rect) )    
      o_rois.append( i_rect )    
                  
    
  #o_rois.sort(key=lambda tup: tup[0], reverse = True)  # sorts in place
  return o_rois  
  
def visualize_rois(size_x, size_y, rois, filename):
  img = np.zeros( (size_y, size_x), dtype=np.uint8)
  
  #max_scale = 0
  #for scale, rect in rois:    
  #  if (scale>max_scale):
  #    max_scale = scale
  
  #for scale, rect in rois:    
  for rect in rois:    
     
    minx, miny, maxx, maxy = rect
    #color = int(255-scale*poznanroi_config_codec.GAUSS_LEVELS_MUL_FOR_ENERGY)
    color = 255
    img[miny:maxy, minx:maxx] = color
    
    #cv.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),  (255,255,255),3)
   
  cv2.imwrite( filename, img.astype(np.uint8) )
