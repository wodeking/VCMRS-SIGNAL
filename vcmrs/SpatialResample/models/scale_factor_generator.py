# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import os
import torch
import math
import cv2
import numpy as np
import pandas as pd
from vcmrs.Utils import data_utils
from vcmrs.InnerCodec.Utils import video_utils
import glob
from types import SimpleNamespace
import vcmrs
class AdaptiveScaleFactorGenerator():
  def __init__(self, item, input_fname):
    if item.args.SpatialDescriptorMode=='NoDescriptor' or item.args.SpatialDescriptorMode=='GeneratingDescriptor':
      self.model = self.setModel()  
        
    self.scale_factor_id_mapping = {
      1.0 : 0,
      0.90 : 1,
      0.70 : 2,
      0.50 : 3,
      0.30 : 4
    }
    
    self.id_scale_factor_mapping = dict(reversed(x) for x in self.scale_factor_id_mapping.items())

    self.corr_thresh = 0.9
    self.top_rank_rate=0.7

    self.cs_error = 0
    self.cr_error = 0
    self.video_info = video_utils.get_video_info(input_fname, item.args)
    if item.args.SourceWidth is None: item.args.SourceWidth = self.video_info.resolution[1]
    if item.args.SourceHeight is None: item.args.SourceHeight = self.video_info.resolution[0]

  def setModel(self):
    # Load Yolov7 model
    spatial_model_dir = os.path.dirname(os.path.abspath(__file__))
    weight = f"{spatial_model_dir}/weights/yolov7.pt"   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(f"{spatial_model_dir}/yolov7","custom",weight,source='local', verbose=False)
    model.to(device)
    return model

  def calculate_correlation(self, hist_1, hist_2):
    h1 = hist_1.reshape(len(hist_1),1).astype(np.float32) 
    h2 = hist_2.reshape(len(hist_2),1).astype(np.float32) 

    v1 = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return v1

  def generate_scale_factor_list(self, item, pre_analyzed_data):

    chunk_info_list = self.chunk_scale_list(item)
    vcmrs.log(f'viscom debug: \n {chunk_info_list}')
    for chunk_info in chunk_info_list :
      scale_list=[]
      for frameNum in range(chunk_info.start_idx, chunk_info.end_idx + 1):
        scale = list(self.scale_factor_id_mapping.keys())
        frame_data = pre_analyzed_data[pre_analyzed_data['frameNum'] == frameNum]
        
        # Exeption: no detection results for all
        if len(frame_data)<1:
          self.cs_error += 1
          # vcmrs.log('No datas', self.cs_error) 
          scale_list.append(self.scale_factor_id_mapping[scale[-1]])
        # Exeption: no detection results in original
        elif (processed_data := self.set_object_information(item, frame_data)) is None:
          scale_list.append(self.scale_factor_id_mapping[scale[-1]])
        # for normal
        else:     
          scale_list.append(self.get_scalefactors(processed_data, item, frameNum)) 
          
      #self.selective_deactivation(item, 0.25)
      self.save_scale_list(scale_list, item, chunk_info.chunk_idx)
      # it is needed to implement that the scale list is saved in memory instead of files.
      
  def get_scalefactors(self, data, item, frameNum):
    scale = list(self.scale_factor_id_mapping.keys())
    rc_scale = 0
    if len(data)<1:
        self.cs_error += 1
        return self.scale_factor_id_mapping[scale[-1]]

    oar_category=round(data['oar_category'].mean())
    if oar_category==1:
        xbin = [x for x in np.arange(0, 1.0, 0.01)]
    elif oar_category == 2:
        xbin = [x for x in np.arange(0, 1.0, 0.05)]
    elif oar_category > 2:        
        xbin = [x for x in np.arange(0, 1.0, 0.1)]
    
    anchor, _ = np.histogram(data[data['scale']==1.0]['obj_area_ratio'], bins=xbin) 
    start_idx = 1   
    correlation = []     
    for sc in scale[start_idx:]:    
        hist, _ = np.histogram(data[data['scale']==sc]['obj_area_ratio'], bins = xbin)  
        res = self.calculate_correlation(anchor,hist)      
        correlation.append(res)

    similar=[]
    for x in correlation:
      if x >= self.corr_thresh:  
        similar.append(x)
      else:
        break
      
    if similar:
        rc_scale = self.scale_factor_id_mapping[scale[correlation.index(similar[-1])+1]]
    else:
        self.cr_error += 1
        rc_scale = self.scale_factor_id_mapping[scale[-1]]
    
    rc_scale += self.adjustment_scale_factor(item.args.quality, rc_scale, oar_category, item.args.SourceWidth)
    if rc_scale <= 0 : rc_scale = 1

    return rc_scale

  def adjustment_scale_factor(self, qp, scale, oar_category, res):
    quality_bias = [4, 6, 8]
    qp_basis = [25, 30, 35, 40, 45]
    qp_score = oar_score = resol_score = 0
    
    # adjust recomended scale by QP  
    if qp < qp_basis[0]:  qp_score = 0 
    elif qp < qp_basis[1]: qp_score = 1
    elif qp < qp_basis[2]:  qp_score = 2
    elif qp < qp_basis[3]:  qp_score = 3
    elif qp < qp_basis[4]:  qp_score = 4
    else :  qp_score = 5
    
    # adjust recomended scale by OAR
    # if scale == list(self.scale_factor_id_mapping.values())[-1] and oar_category == 1:
    if self.id_scale_factor_mapping[scale] <= 0.5 and oar_category == 1 :  oar_score = 3
    
    if res < 640 : resol_score = 2
    
    quality_score = qp_score + scale + oar_score + resol_score
    
    if quality_score > quality_bias[2]: return -3
    elif quality_score > quality_bias[1]: return -2
    elif quality_score > quality_bias[0]: return -1
    else: return 0

  def read_frame(self, input_fname, frameNum, item):
    
    # dir(images) -> frame  
    if item._is_video and not item._is_yuv_video: # video as directory (pngs)
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      img_bgr = cv2.imread(fnames[frameNum])
      frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    elif os.path.isfile(input_fname): # YUV file or image data      
      # yuv, frameNum -> frame
      if item._is_yuv_video :
        frame_yuv = data_utils.get_frame_from_raw_video(input_fname, frameNum, bit_depth=item.args.InputBitDepth, W=item.args.SourceWidth, H=item.args.SourceHeight) 
        frame = data_utils.cvt_yuv444p_to_rgb(frame_yuv, item.args.InputBitDepth)
      # img -> frame
      else: # frame data
        img_bgr = cv2.imread(input_fname)
        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
      raise FileNotFoundError(f"Input {input_fname} is not found.")
  
    return frame

  def get_object_information(self, item, input_fname):
    obj_infos = pd.DataFrame()
    # for frameNum in range(video_info.num_frames):  
    #   if frameNum >= item.args.FrameSkip and frameNum < (item.args.FrameSkip + item.args.FramesToBeEncoded):   
    start = item.args.FrameSkip
    if item.args.FramesToBeEncoded == 0:
      end = self.video_info.num_frames
    else:
      end = item.args.FrameSkip + item.args.FramesToBeEncoded
    for frameNum in range(start, end):
      img = self.read_frame(input_fname, frameNum, item)
      obj_infos=pd.concat([obj_infos,
                           self.get_object_information_per_frame(img, item, frameNum)])
    
    return obj_infos

  def get_object_information_per_frame(self, image, item, frameNum):
    scale = list(self.scale_factor_id_mapping.keys())
    
    for sc in scale :
      if 'result_all_sc' in locals():
        result_all_sc = pd.concat([result_all_sc, self.get_object_information_per_scale(sc, image, item, frameNum)])
      else :
        result_all_sc = self.get_object_information_per_scale(sc, image, item, frameNum)      
    return result_all_sc

  def get_object_information_per_scale(self, sc, image, item, frameNum):
    scaled_wdt = int(math.ceil(((item.args.SourceWidth*sc)/8))*8)
    scaled_hgt = int(math.ceil(((item.args.SourceHeight*sc)/8))*8)  
    resized_img = cv2.resize(image, (scaled_wdt, scaled_hgt), interpolation = cv2.INTER_CUBIC) 
    reconstructed_img = cv2.resize(resized_img, (item.args.SourceWidth, item.args.SourceHeight), interpolation = cv2.INTER_CUBIC)
    results = self.model(reconstructed_img).pandas().xyxy[0]
    results['w'] = results['xmax']-results['xmin']
    results['h'] = results['ymax']-results['ymin']
    results['frameNum'] = frameNum
    results['scale'] = sc
    results['filename'] = item._bname
    results.reset_index(inplace=True)
    results['obj_idx'] = results['index'].astype(int)
    results = results[['filename','frameNum','obj_idx','xmin','ymin','xmax','ymax','confidence','class','name','w','h','scale']]
    results = results.astype({'scale':'float'})
    return results
  
  def set_object_information(self, item, data):
    data['obj_area_ratio']=(data['w']*data['h'])/(item.args.SourceWidth*item.args.SourceHeight)

    data['oar_category']=0
    data.loc[data['obj_area_ratio'] < 0.01, 'oar_category'] = 1
    data.loc[(data['obj_area_ratio'] >= 0.01) & (data['obj_area_ratio'] < 0.10), 'oar_category'] = 2
    data.loc[(data['obj_area_ratio'] >= 0.10) & (data['obj_area_ratio'] < 0.20), 'oar_category'] = 3
    data.loc[(data['obj_area_ratio'] >= 0.20) & (data['obj_area_ratio'] < 0.30), 'oar_category'] = 4
    data.loc[data['obj_area_ratio'] >= 0.30, 'oar_category'] = 5

    original_data = data[data['scale']==1.0].sort_values(by='confidence', ascending=False)
    original_data.reset_index(inplace=True)

    if original_data.empty:
      return None
    else:
      confidence_thresh = original_data.loc[int(math.ceil(len(original_data) * self.top_rank_rate))-1]['confidence']
      new_data = data[data['confidence']>round(confidence_thresh, 2)]
    return new_data
  
  def selective_deactivation(self, item, threshold):    
    if self.cs_error > item.args.FramesToBeEncoded * threshold:       item.args.SpatialResample =  'Bypass'
    else:                                                   item.args.SpatialResample = 'AdaptiveDownsample'
    
  def save_scale_list(self, scale_list, item, chunk_idx):
    scale_list = '\n'.join(list(map(str, scale_list))) + '\n'
    
    scaleListFileDir = os.path.join(item.args.working_dir, 'scalelist')
    if not os.path.exists(scaleListFileDir): 
      os.makedirs(scaleListFileDir) 
    scale_list_file = os.path.join(scaleListFileDir, f'{item.args.working_dir.split(os.path.sep)[-1]}_{chunk_idx}.txt')
    
    with open(scale_list_file, 'w+') as f:
        f.write(scale_list) 
  
  def read_descriptor(self, path):
    data = pd.read_csv(path)
    return data
    
  def chunk_scale_list(self, item):
    num_frames = self.video_info.num_frames - item.args.FrameSkip
    if item.args.FramesToBeEncoded > 0: num_frames = min(num_frames, item.args.FramesToBeEncoded)

    if item.args.single_chunk:
      chunk_size = num_frames
    else:
      chunk_size = item.IntraPeriod

    video_start_idx = item.args.FrameSkip
    video_end_idx = video_start_idx + num_frames - 1 # inclusive

    # prepare chunk information
    chunk_info_list=[]
    n_chunks = math.ceil(num_frames / chunk_size)
    for chunk_idx in range(n_chunks):
      chunk_info = SimpleNamespace()
      chunk_info.chunk_idx = chunk_idx
      chunk_info.start_idx = video_start_idx + chunk_idx * chunk_size
      chunk_info.end_idx = min(chunk_info.start_idx + chunk_size, video_end_idx)
      chunk_info.intra_period = item.IntraPeriod
      chunk_info_list.append(chunk_info)    
      
      
    return chunk_info_list