# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import vcmrs
from numba import jit
from collections import deque
import torch
import sys
import re
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.kalman_filter import KalmanFilter
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.log import logger
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.models import *
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.tracker import matching
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.tracker.basetrack import BaseTrack, TrackState
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils import datasets as datasets
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.parse_config import parse_model_cfg
from vcmrs.ROI.roi_retargeting import roi_retargeting
from vcmrs.ROI.roi_utils import roi_utils
import os
import zlib
import base64
import json
import subprocess
import shutil
import cv2
import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from vcmrs.Utils import utils



class RoIImageGenerator(object):
  def __init__(self, opt, item, seqLength=0):
    self.opt = opt
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vcmrs.log(f"RoI generation with {self.device}")
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    if item.args.RoIGenerationNetwork == 'yolov3_1088x608':
      cfg_dict = parse_model_cfg(opt.cfg)
      self.img_size = [int(cfg_dict[0]["width"]), int(cfg_dict[0]["height"])]
      #dataloader = datasets.LoadImages(input_dir, self.img_size)
      #self.generate_RoI_YOLOv4(input_dir, item, dataloader, output_dir, self.accumulation_period)

      self.model = Darknet(opt.cfg, nID=14455)
      self.model.load_state_dict(torch.load(opt.weights, map_location=self.device)['model'], strict=False)
      self.model.to(self.device).eval()
    elif item.args.RoIGenerationNetwork == 'faster_rcnn_X_101_32x8d_FPN_3x':
      # setup detectron2 logger
      setup_logger()

      # create config
      cfg = get_cfg()
      
      cfg.merge_from_file(os.path.join(cur_dir_path, "./config/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      cfg.MODEL.WEIGHTS = os.path.join(cur_dir_path, "./weights/model_final_68b088.pkl")
      if self.device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu'
      # create predictor
      self.predictor = DefaultPredictor(cfg)
    
    self.item=item
    self.frame_id = 0
    self.objects_for_frames = []
    self.img_temp = []
    self.seqLength = seqLength
    self.descriptor = []
    self.frame_skip = 0 #item.args.FrameSkip  # instead of preserving FrameSkip - remove those frames, and set FrameSkip=0 (required for potential change of order between TR and ROI)
    
    
  def makeBinaryMask(self, img, input_dict, width, height):    
    objects = []
        
    for i in range(len(input_dict)):
      box = input_dict[i].cpu().numpy()
      x1 = int(box[0])
      x2 = int(box[2])
      y1 = int(box[1])
      y2 = int(box[3])
      objects.append([x1,y1,x2,y2])
      
    objects.sort(key = lambda x:(x[0], x[1]))
    widthThreshold = width/60
    heightThreshold = height/25
    
    for i in range(len(objects)-1):
      for j in range(i,len(objects)-i-1):
        if(abs(objects[j][2]-objects[j+1][0]) < widthThreshold or objects[j][2] > objects[j+1][0]):
          if(abs(objects[j][1]-objects[j+1][1]) < heightThreshold and abs(objects[j][3] - objects[j+1][3]) < heightThreshold):
            objects[j+1][0] = objects[j][0]
            objects[j+1][1] = objects[j][1] if objects[j][1] < objects[j+1][1] else objects[j+1][1]
            objects[j+1][3] = objects[j][3] if objects[j][3] > objects[j+1][3] else objects[j+1][3]
     
    return objects

  def roiAccumulation(self, img, width, height, start_idx, end_idx): # equivalent of original proposal, code refactored
    binary_FG = np.full((height, width,3),127,dtype=np.uint8)  
    feather = 20

    for j in range(start_idx, end_idx+1):
      for i in range(len(self.objects_for_frames[j])):
        bbox = self.objects_for_frames[j][i]
        x1, y1, x2, y2 = roi_utils.rect_extend_limited(bbox, feather, width, height)
      
        binary_FG[y1:y2, x1:x2,:] = img[y1:y2, x1:x2,:]
        
    return binary_FG
  
  def getMaskWithFeather(self, width, height, feather, value, start_idx, end_idx):
    return roi_utils.get_mask_with_feather(self.objects_for_frames, width, height, feather, value, start_idx, end_idx)

  def roiAccumulationRetargeting(self, img, retargeting_params, width, height, start_idx, end_idx):

    org_size_y, org_size_x, num_comps = img.shape
    
    org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = retargeting_params

    #deep_background_mode = "before_rtg"
    deep_background_mode = "after_rtg_64"

    feather = 10
    slope_to_blur_size_half = 50 # feather*3        slope_to_blu_size_total = 120
    closing = 160
    deep_background_color = 127

    mask_i = self.getMaskWithFeather(width, height, 5, 1, start_idx, end_idx)
    
    scale = 5
    blur = roi_utils.blur_background_scaled(101, mask_i, img, 0.001*0.5*scale, scale)

    if deep_background_color=="average":    
      #deep_background_color = blur.mean(axis=0).mean(axis=0) # not numerically stable
      dy, dx, comps = blur.shape
      num_pixels = dx*dy
      deep_background_color = np.zeros( (comps), dtype = np.uint8)
      for c in range(comps):
        deep_background_color[c] = np.clip( (blur[:,:,c].sum(dtype=np.int64) + num_pixels//2) // num_pixels, 0,255)
    
    mask_slope = self.getMaskWithFeather(width, height, feather, 128, start_idx, end_idx)
    mix_slope = roi_utils.mix_images(blur, mask_slope, slope_to_blur_size_half, img)
    mask_closing = self.getMaskWithFeather(width, height, feather+closing, 128, start_idx, end_idx)
    mask_closing = roi_utils.erode_image(mask_closing, closing*2+1)
    
    if deep_background_mode == "before_rtg":
      deep_background = np.zeros((height, width,3),dtype=np.uint8)  
      deep_background[:,:,:] = deep_background_color
    
      img = roi_utils.mix_images(deep_background, mask_closing, 0, mix_slope)
    else: 
      img = mix_slope

    img = roi_retargeting.retarget_image(img, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y)
    
    if deep_background_mode.startswith("after_rtg"):      
      
      mask_closing3 = np.zeros( (org_size_y, org_size_x, num_comps), dtype = np.uint8)
      mask_closing3[:,:,0] = mask_closing
      
      mask_closing3_rtg = roi_retargeting.retarget_image(mask_closing3, org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y)
            
      mask_closing_rtg = mask_closing3_rtg[:,:,0]
      
      spl = deep_background_mode.split("_")
      raster = int(spl[2])
      
      dx = rtg_size_x
      dy = rtg_size_y
      
      dxn = (dx+raster-1)//raster
      dyn = (dy+raster-1)//raster
      
      dxr = dxn * raster
      dyr = dyn * raster
      
      if (dx!=dxr) or (dy!=dyr):
        tmp = np.zeros( (dyr, dxr), dtype = np.uint8)
        tmp[0:dy, 0:dx] = mask_closing_rtg
 
        for x in range(dx, dxr):
          tmp[0:dy, x] = mask_closing_rtg[0:dy, dx-1]
            
        for y in range(dy, dyr):
          tmp[y, 0:dx] = mask_closing_rtg[dy-1, 0:dx]
                
        tmp[dy:, dx:] = mask_closing_rtg[dy-1, dx-1]
        
        mask_closing_rtg = tmp
            
      mask_closing_rtgs = cv2.resize(mask_closing_rtg, (dxn, dyn), interpolation=cv2.INTER_AREA)
      
      #mask_closing_rtgst = (mask_closing_rtgs > 64).astype(np.uint8)*128
      mask_closing_rtgst = (mask_closing_rtgs >= 2).astype(np.uint8)*128   # 1.5%
      
      mask_closing_rtgstr = cv2.resize(mask_closing_rtgst, (dxr, dyr), interpolation=cv2.INTER_AREA)
      mask_closing_rtgstr = mask_closing_rtgstr[0: rtg_size_y, 0:rtg_size_x]
      
      deep_background = np.zeros((rtg_size_y, rtg_size_x,3),dtype=np.uint8)  
      deep_background[:,:,:] = deep_background_color
      
      img = roi_utils.mix_images(deep_background, mask_closing_rtgstr, 0, img)

    return img

  def networkDetectRoI(self, img_rgb):
    
    if self.item.args.RoIGenerationNetwork == 'yolov3_1088x608':  
      
      # Padded resize
      im_blob, _, _, _ = datasets.letterbox(img_rgb, height=self.img_size[1], width=self.img_size[0])

      # Normalize RGB
      im_blob = im_blob[:, :, ::-1].transpose(2, 0, 1)
      im_blob = np.ascontiguousarray(im_blob, dtype=np.float32)
      im_blob /= 255.0
      
      im_blob = torch.from_numpy(im_blob).to(self.opt.device).unsqueeze(0)
    
      with torch.no_grad():
        pred = self.model(im_blob)

      pred = pred[pred[:, :, 4] > self.opt.conf_thres]

      if len(pred) > 0:
        dets1 = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].to(self.device)

        scale_coords(self.img_size, dets1[:, :4], img_rgb.shape).round()

        det2 = [tlbrs[:4] for (tlbrs, f) in zip(dets1[:, :5], dets1[:, 6:])]
      
        return RoIImageGenerator.makeBinaryMask(self, img_rgb, det2, img_rgb.shape[1], img_rgb.shape[0])
    
    elif self.item.args.RoIGenerationNetwork == 'faster_rcnn_X_101_32x8d_FPN_3x':
      # read image

      # make prediction
      outputs = self.predictor(img_rgb)
      
      # get predictions
      instances = outputs["instances"]
      det2 = instances.pred_boxes.tensor
    
      return RoIImageGenerator.makeBinaryMask(self, img_rgb, det2, img_rgb.shape[1], img_rgb.shape[0])
    
    return [[0,0,0,0,]]
    
          
  def generateRoIImage(self, roi_update_period, accumulation_period, desired_max_obj_size, max_num_rois, img_rgb, save_name_template):
    if self.frame_id % 20 == 0:
      vcmrs.log(f"RoI processing: frame {self.frame_id} accumulation_period{accumulation_period} roi_update_period{roi_update_period}")

    if self.item.args.RoIDescriptorMode=="load" and (self.item.args.RoIDescriptor is not None):
      mode_network_generate_descriptor = False
      mode_load_descriptor_from_file   = True
      mode_save_descriptor_to_file     = False
      mode_exit_after_roi              = False
    elif (self.item.args.RoIGenerationNetwork is not None):     
      mode_network_generate_descriptor = True
      mode_load_descriptor_from_file   = False
      mode_save_descriptor_to_file     = (self.item.args.RoIDescriptorMode=="save") or (self.item.args.RoIDescriptorMode=="saveexit")
      mode_exit_after_roi              = (self.item.args.RoIDescriptorMode=="saveexit")
    else:
      vcmrs.log(f"RoI processing: no source of descriptors! Specify RoIGenerationNetwork or RoIDescriptor parameter!")
      sys.exit(0)

    if mode_network_generate_descriptor:
      # use networks to generate
      objects = self.networkDetectRoI(img_rgb)
      self.objects_for_frames.append(objects)

    img_process = img_rgb
    self.img_temp.append(img_process)
    
    # wait until end of sequence and process img_temp
    #if self.frame_id == 1: # second frame, for testing
    if self.frame_id == self.seqLength-1:
    
      if mode_load_descriptor_from_file:
        vcmrs.log(f"RoI processing: loading descriptor {self.item.args.RoIDescriptor}")
        self.objects_for_frames = roi_utils.load_descriptor_from_file( self.item.args.RoIDescriptor )

      if mode_save_descriptor_to_file:
        descriptor_file = self.item.args.RoIDescriptor
        if descriptor_file is None:
          descriptor_path = os.path.join(self.item.working_dir, 'RoIDescriptor')
          if not os.path.exists(descriptor_path):
            os.makedirs(descriptor_path, exist_ok=True)
          descriptor_file = os.path.basename(self.item.fname) + ".txt"
          descriptor_file = re.sub(r'_qp\d+', '', descriptor_file)
          descriptor_file = os.path.join(descriptor_path,descriptor_file)
            
        if os.path.isfile(descriptor_file):
          for i in range(1,1000):
            test = descriptor_file+"("+str(i)+")"
            if not os.path.isfile(test):
              break
          descriptor_file = test
            
        vcmrs.log(f"RoI processing: saving descriptor {descriptor_file}")
        
        # save... (and exit later maybe?)
        roi_utils.save_descriptor_to_file(descriptor_file, self.objects_for_frames)

      if mode_exit_after_roi:
        vcmrs.log(f"RoI processing: exitting after saving")
        sys.exit(0)
    
      vcmrs.log(f"RoI processing: saving images")

      width, height = img_process.shape[1], img_process.shape[0] # last frame in the sequence

      retargeting_enabled = self.item.args.RoIRetargetingMode != "off"
      #retargeting_enabled = False
      
      retargeting_decision_sequence = self.item.args.RoIRetargetingMode == "sequence"

      if retargeting_enabled:
        
        num_roi_update_periods =  (self.seqLength + roi_update_period -1 ) // roi_update_period # round up
        ip_rois = []
        
        #feather = 20
        feather = 5

        for ip in range(num_roi_update_periods):
          start_idx = ip * roi_update_period
          end_idx = min(start_idx + roi_update_period, self.seqLength)-1

          rois = roi_retargeting.find_ROIs(self.objects_for_frames, feather, desired_max_obj_size, width, height, start_idx, end_idx, None)

          if len(rois)>max_num_rois:
            rois = roi_retargeting.find_ROIs_simple(self.objects_for_frames, feather, width, height, start_idx, end_idx)

          ip_rois.append( rois )
        
        align_size = 64
        rtg_size_x_final = 0
        rtg_size_y_final = 0
        for ip in range(num_roi_update_periods):
          org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y =  roi_retargeting.generate_coords(width, height, ip_rois[ip], align_size, align_size, None, None)
          vcmrs.log(f"RoI processing: ip: {ip:02d} would suggest resolution:    {width}x{height} -> {rtg_size_x}x{rtg_size_y}")
          
          if (ip==0) or (retargeting_decision_sequence): 
            rtg_size_x_final = max(rtg_size_x_final, rtg_size_x)
            rtg_size_y_final = max(rtg_size_y_final, rtg_size_y)
                   
        vcmrs.log(f"RoI processing: retargeting: Resolution suggested:  {width}x{height} -> {rtg_size_x_final}x{rtg_size_y_final}")
        
        rtg_size_x_final = max(rtg_size_x_final, width//16) # do not shrink more than 1:16
        rtg_size_y_final = max(rtg_size_y_final, height//16)
        
        rtg_size_x_final = max(rtg_size_x_final, 1) # do not shrink more than to 1x1 pixels
        rtg_size_y_final = max(rtg_size_y_final, 1)
        
        rtg_size_x_final = (rtg_size_x_final+1) & ~1 # even number of pixels (4:2:0)
        rtg_size_y_final = (rtg_size_y_final+1) & ~1 # even number of pixels (4:2:0)
        
        rtg_size_x_final = min(rtg_size_x_final, width) # do not extend (possible due to alignment, etc.)
        rtg_size_y_final = min(rtg_size_y_final, height)
        
        rtg_size_x_final = max(rtg_size_x_final, 128) # for LIC intra codec
        rtg_size_y_final = max(rtg_size_y_final, 128)
        
        vcmrs.log(f"RoI processing: retargeting: Resolution final:      {width}x{height} -> {rtg_size_x_final}x{rtg_size_y_final}")

        retargeting_params_for_ip = []
        for ip in range(num_roi_update_periods):
          res = roi_retargeting.generate_coords(width, height, ip_rois[ip], None, None, rtg_size_x_final, rtg_size_y_final)
          retargeting_params_for_ip.append(res)
      
        num_accumulation_periods = (self.seqLength + accumulation_period -1 ) // accumulation_period # round up
        for ap in range(num_accumulation_periods):
          start_idx = ap * accumulation_period
          end_idx = min(start_idx + accumulation_period, self.seqLength)-1
          for f in range(start_idx, end_idx+1):
            img = self.roiAccumulationRetargeting(self.img_temp[f], retargeting_params_for_ip[f // roi_update_period], width, height, start_idx, end_idx)
            cv2.imwrite(save_name_template % (f+self.frame_skip),img) 

      else: # no retargetting - Original Code (refactored)

        if self.seqLength <= accumulation_period:
          for i in range(self.seqLength):
            img_save = RoIImageGenerator.roiAccumulation(self, self.img_temp[i], width, height, 0, self.seqLength-1)
            cv2.imwrite( save_name_template % (i+self.frame_skip) ,img_save) 
        else:
          for i in range(self.seqLength):
            if (i+1) % accumulation_period == 0:
              if (i+accumulation_period) > (self.seqLength-1):
                for j in range(i-(accumulation_period-1),i+1):
                  img_save = RoIImageGenerator.roiAccumulation(self, self.img_temp[j], width, height, i-(accumulation_period-1), i)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save) 
                for j in range(i+1,self.seqLength):
                  img_save = RoIImageGenerator.roiAccumulation(self, self.img_temp[j], width, height, i, self.seqLength-1)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save)
              else:
                for j in range(i-(accumulation_period-1),i+1):
                  img_save = RoIImageGenerator.roiAccumulation(self, self.img_temp[j], width, height, i-(accumulation_period-1), i)
                  cv2.imwrite( save_name_template % (j+self.frame_skip),img_save) 


      vcmrs.log(f"RoI processing: done")
      
      if retargeting_enabled:
        return rtg_size_x_final, rtg_size_y_final,     rtg_size_x_final, rtg_size_y_final,     ip_rois
      return None, None,     None, None,     None
      
    self.frame_id += 1
    return None, None,     None, None,     None
  
