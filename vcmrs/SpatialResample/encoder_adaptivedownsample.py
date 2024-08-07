# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os, sys
import cv2
import vcmrs
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
import re

import numpy as np
import pandas as pd
import math
import torch
# import warnings # disable warnings

from pathlib import Path
import vcmrs.SpatialResample.models as spatial_model
from vcmrs.SpatialResample.models import scale_factor_generator 
from vcmrs.Utils.io_utils import enforce_symlink 

class AdaptiveDownsample(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
    
    vcmrs.log("Spatial Resampling : ===================== Calculate adaptive scaling factors ====================")
    
    self.generator = scale_factor_generator.AdaptiveScaleFactorGenerator(item, input_fname)
    
    if item.args.SpatialDescriptorMode=='GeneratingDescriptor' or item.args.SpatialDescriptorMode=='UsingDescriptor':
      if (descriptor_file := item.args.SpatialDescriptor) is None:
        descriptor_file = os.path.join(os.path.dirname(os.path.dirname(vcmrs.__file__)), "Data", "spatial_descriptors",  os.path.basename(item.args.output_recon_fname) + ".csv")
        descriptor_file = re.sub(r'_qp\d+', '', descriptor_file)
            
      descriptor_dir = os.path.dirname(descriptor_file)
        
    if item.args.SpatialDescriptorMode=='NoDescriptor': 
      object_info = self.generator.get_object_information(item, input_fname)
      if object_info.empty : 
        item.args.SpatialResample =  'Bypass'
        enforce_symlink(input_fname, output_fname)
        return -1
      self.generator.generate_scale_factor_list(item, object_info)
      # optimal_scale_factor_list = generate_scale_factor_list(item, pre_analyzed_data)
      
    elif item.args.SpatialDescriptorMode=='GeneratingDescriptor':
      object_info = self.generator.get_object_information(item, input_fname)
      
      vcmrs.log("Spatial Resampling : start to write the csv file")
      vcmrs.log(f"Spatial Resampling : write the {descriptor_file}")
      
      if os.path.exists(descriptor_file) :
        vcmrs.log("Spatial Resampling : Error : descriptor_file is already exist. Failed to generate the descriptor file. ")
        sys.exit(0)
      
      os.makedirs(descriptor_dir, exist_ok=True)
      
      object_info.to_csv(descriptor_file, header=True, mode='w', sep=',', index=False)

      vcmrs.log("Spatial Resampling : Done : Generating the spatial descriptor is completed.")
      sys.exit(0)
      
    elif item.args.SpatialDescriptorMode=='UsingDescriptor':
      if not os.path.exists(descriptor_file):
        vcmrs.log("Spatial Resampling : Error : descriptor_path is not exist.")
        sys.exit(0)
            
      object_info = self.generator.read_descriptor(descriptor_file)
      self.generator.generate_scale_factor_list(item, object_info)

                
    enforce_symlink(input_fname, output_fname)
    
    # Syntax for spatial resampling
    vcm_based_upscaling_flag = 0
    self._set_parameter(item, vcm_based_upscaling_flag)
    
    vcmrs.log("Spatial Resampling : ================================= complete  =================================")
    return 0

  def _set_parameter(self, item, vcm_based_upscaling_flag=0):
    # sequence level parameter
    param_data = bytearray(1)
    param_data[0] = vcm_based_upscaling_flag
    item.add_parameter('SpatialResample', param_data=param_data)
    pass