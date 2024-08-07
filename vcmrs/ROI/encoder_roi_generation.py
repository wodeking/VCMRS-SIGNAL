# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

import os
import atexit
import cv2
import shutil
import torch
import shutil
import vcmrs
from vcmrs.ROI.roi_generator import roi_accumulation as roi_accumulation
from vcmrs.ROI.roi_syntax import roi_bitenc
import subprocess
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils import datasets as datasets
from vcmrs.ROI.roi_generator.Towards_Realtime_MOT.utils.utils import *
from vcmrs.Utils.component import Component
from vcmrs.Utils import io_utils
from vcmrs.Utils import data_utils
from vcmrs.Utils import utils

# component base class
class roi_generation(Component):
  def __init__(self, ctx):
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.ctx = ctx
    self.cfg = os.path.join(cur_dir_path,"./roi_generator/Towards_Realtime_MOT/cfg/yolov3_1088x608.cfg")
    self.weights = os.path.join(cur_dir_path, "./roi_generator/jde.1088x608.uncertainty.pt")
    self.iou_thres = 0.5
    self.conf_thres = 0.5
    self.nms_thres = 0.4
    self.accumulation_period = 64
    self.roi_update_period = 1
    self.desired_max_obj_size = 200
    self.max_num_rois = 11
    self.track_buffer = 30
    self.img_size = [0, 0]
    self.working_dir = ''
    atexit.register(self.cleanup)
    
  def cleanup(self):
    if self.working_dir != '':
      yuv_temp = os.path.abspath(os.path.join(self.working_dir,"yuv_temp"))
      temp = os.path.abspath(os.path.join(self.working_dir,"temp"))
      if os.path.exists(yuv_temp):
        shutil.rmtree(yuv_temp)
      if os.path.exists(temp):
        shutil.rmtree(temp)
    
    
  def _is_image(self, file_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.svg', '.tiff', '.ico']
    ext = os.path.splitext(file_path)[1]
    return ext.lower() in img_extensions
  
  # generate_RoI_YOLOv4 moved to unified networkDetectRoI function

  # generate_RoI_FasterRCNN moved to unified networkDetectRoI function

  def generateRoI(self, input_dir, output_dir, item):
    # specify the path to the folder containing .png files
    temp_output_path = io_utils.create_temp_folder_suffix( os.path.abspath(os.path.join(item.args.working_dir,"yuv_temp")) )

    # get a list of all files in the folder
    png_list = data_utils.get_image_file_list(input_dir)

    input_frame_skip = item.args.FrameSkip # using frame_skip only on the input side.
    frames_to_be_encoded = item.args.FramesToBeEncoded
    seqLength = frames_to_be_encoded
    if seqLength==0:
      seqLength=len(png_list)

    generator = roi_accumulation.RoIImageGenerator(
      opt=self,
      item=item,
      seqLength=seqLength,
    )

    for i in range(len(png_list)):
      if i >= item.args.FrameSkip and i < (input_frame_skip + seqLength):
        image_path = os.path.join(os.path.abspath(input_dir), png_list[i])
        img_rgb = cv2.imread(image_path)

        org_size_y, org_size_x, _ = img_rgb.shape

        update_size_x, update_size_y, rtg_size_x, rtg_size_y, retargeting_gops_rois = generator.generateRoIImage(self.roi_update_period, self.accumulation_period, self.desired_max_obj_size, self.max_num_rois, img_rgb, output_dir)
        
        # if retargettig is enabled this is returned as not None
        if retargeting_gops_rois is not None:
          gops_roi_bytes = roi_bitenc.encode_gops_rois_params(self.roi_update_period, org_size_x, org_size_y, rtg_size_x, rtg_size_y, retargeting_gops_rois, vcmrs.log)
          item.add_parameter('ROI', param_data=bytearray(gops_roi_bytes) ) 
        
        # update inner codec resolution conly if padding is disabled  
        if update_size_x is not None:
          item.args.SourceWidth = update_size_x
        if update_size_y is not None:
          item.args.SourceHeight = update_size_y

      #else:
      #  symlink_path = os.path.join(
      #    temp_output_path, "frame_{:06d}.png".format(i)
      #  )
      #  io_utils.enforce_symlink(image_path, symlink_path)
      # instead of preserving FrameSkip - remove those frames, and set FrameSkip=0 (required for potential change of order between TR and ROI)

    item.args.FrameSkip = 0 # set current FrameSkip to 0, as on the output it is not used
    
  def process(self, input_fname, output_fname, item, ctx):
    self.working_dir = os.path.abspath(item.args.working_dir)
    
    if item.args.RoIAccumulationPeriod == 0: # default
      if item.args.Configuration == 'AllIntra':
        self.accumulation_period = 1
      elif item.args.Configuration == 'LowDelay':
        self.accumulation_period = 32 # defaults to CTC configuration which is 32 frames, indepedently from temporal resampling, also as originally in ROI plugin
      else:                
        self.accumulation_period = item.IntraPeriod # intra period, but adjusted to operation of temporal resampling (if active)
        
    elif item.args.RoIAccumulationPeriod<0: # negative: take value as absolute, not adjusted
      self.accumulation_period = abs(item.args.RoIAccumulationPeriod) 
    else: # postive: value, but adjusted to operation of temporal resampling (if active)
      self.accumulation_period = int( abs(item.args.RoIAccumulationPeriod) * item.FrameRateRelativeVsInput )
      
    self.accumulation_period = max(self.accumulation_period, 1) # must be >= 1

    self.roi_update_period = min( item.IntraPeriod, self.accumulation_period )
    self.roi_update_period = max(self.roi_update_period, 1) # must be >= 1

    self.desired_max_obj_size = 200
    if item.args.RoIGenerationNetwork=="faster_rcnn_X_101_32x8d_FPN_3x": self.desired_max_obj_size = 100
    if item.args.RoIGenerationNetwork=="yolov3_1088x608": self.desired_max_obj_size = 320 

    self.max_num_rois = item.args.RoIRetargetingMaxNumRoIs

    if item._is_yuv_video:
      height = item.args.SourceHeight
      width = item.args.SourceWidth

      temp_folder_path_rgb = io_utils.create_temp_folder_suffix(os.path.abspath(os.path.join(item.args.working_dir,"temp_rgb")))
      temp_folder_path_out = io_utils.create_temp_folder_suffix(os.path.abspath(os.path.join(item.args.working_dir,"temp_out")))

      rgb_file_name_template  = os.path.join(temp_folder_path_rgb, "frame_%06d.png")
      out_file_name_template  = os.path.join(temp_folder_path_out, "frame_%06d.png")

      pixfmt = "yuv420p" if item.args.InputBitDepth==8 else "yuv420p10le"
      ffmpeg_command = [item.args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error" ]
      ffmpeg_command += ['-threads', '1'] 
      if item.args.InputBitDepth==8:
        ffmpeg_command += ["-f", "rawvideo"]
      ffmpeg_command += [
        "-s", f"{width}x{height}",
        "-pix_fmt", pixfmt,
        "-i", input_fname,
        "-vsync", "1",
        "-y", # duplicate "-y" ????
        "-start_number", "000000",
        "-pix_fmt", "rgb24",
        rgb_file_name_template]
      err = utils.start_process_expect_returncode_0(ffmpeg_command, wait=True)
      assert err==0, "Generating sequence in png format failed."

      self.generateRoI(temp_folder_path_rgb, out_file_name_template, item)

      pixfmt = "yuv420p" if item.args.InputBitDepth==8 else "yuv420p10le"
      ffmpeg_command = [
        item.args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-threads", "1",
        "-f", "image2",
        "-framerate", item.args.FrameRate,
        "-i", out_file_name_template,
        "-pix_fmt", pixfmt,
        output_fname]
      err = utils.start_process_expect_returncode_0(ffmpeg_command, wait=True)
      assert err==0, "Generating sequence to yuv format failed."

      shutil.rmtree(temp_folder_path_rgb)
      shutil.rmtree(temp_folder_path_out)
      
    elif item._is_dir_video:
      os.makedirs(output_fname, exist_ok=True)
      self.generateRoI(input_fname,  os.path.join(output_fname, "frame_%06d.png"), item)
    elif self._is_image(input_fname):
      if os.path.exists(output_fname):
        if os.path.isdir(output_fname):
          shutil.rmtree(output_fname)
        else:
          os.remove(output_fname)
      io_utils.enforce_symlink(input_fname, output_fname)
