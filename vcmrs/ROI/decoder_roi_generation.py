# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import cv2
import vcmrs
import shutil
from vcmrs.Utils.component import Component
from vcmrs.Utils.io_utils import enforce_symlink
from vcmrs.Utils import utils
from vcmrs.ROI.roi_syntax import roi_bitdec
from vcmrs.ROI.roi_retargeting import roi_retargeting


# component base class
class roi_generation(Component):
  def __init__(self, ctx):
    super().__init__(ctx)
    #self.log = print
    self.log = vcmrs.log
    
  def reverseRetargeting(self, inp_dir, out_dir, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois):
  
    if not os.path.isdir(inp_dir):
      assert False, "Input file should be directory!"
    inp_fnames = sorted(glob.glob(os.path.join(inp_dir, '*.png')))
    
    seq_length = len(inp_fnames)
    
    num_roi_update_periods =  (seq_length + roi_update_period -1 ) // roi_update_period # round up
    retargeting_params_for_ip = []
    for ip in range(num_roi_update_periods):
      res = roi_retargeting.generate_coords(org_image_size_x, org_image_size_y, retargeting_gops_rois[ip], None, None, rtg_image_size_x, rtg_image_size_y)
      retargeting_params_for_ip.append(res)
    
    for f in range(seq_length):
      inp_fname = inp_fnames[f]
      basename = os.path.basename(inp_fname)
      out_fname = os.path.join(out_dir, basename)      
      ip = f // roi_update_period
      org_coords_x, org_coords_y, rtg_coords_x, rtg_coords_y, rtg_size_x, rtg_size_y = retargeting_params_for_ip[ip]
      
      img = cv2.imread(inp_fname) # BGR
      assert img is not None, f"Cannot read file:{inp_fname}"
        
      #cv2.imwrite(f"dec{f}a.png", img)
      img = roi_retargeting.retarget_image(img, rtg_coords_x, rtg_coords_y, org_coords_x, org_coords_y)
      cv2.imwrite(out_fname, img)
      #cv2.imwrite(f"dec{f}b.png", img)
      
    return

  def process(self, input_fname, output_fname, item, ctx):
    # the default implementation is a bypass component

    try:
      gops_roi_bytes = item.get_parameter('ROI')
    except:
      gops_roi_bytes = None

    if gops_roi_bytes is not None:
      roi_update_period, rtg_image_size_x, rtg_image_size_y, org_image_size_x, org_image_size_y, retargeting_gops_rois = roi_bitdec.decode_roi_params(gops_roi_bytes, self.log)      
          
      item.args.SourceWidth = org_image_size_x
      item.args.SourceHeight = org_image_size_y
      H,W,C = item.video_info.resolution
      item.video_info.resolution = (org_image_size_y, org_image_size_x, C)
      
      if item._is_dir_video: # input format is directory with separate files           
        
        os.makedirs(output_fname, exist_ok=True)    
        self.reverseRetargeting(input_fname, output_fname, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois)
      elif item._is_yuv_video: # input format is yuv file,  convert to pngs and back
      
        pngpath_before = output_fname+".tmp_dir_in"
        pngpath_after = output_fname+".tmp_dir_out"
        
        os.makedirs(pngpath_before, exist_ok=True)
        os.makedirs(pngpath_after,  exist_ok=True)
        cmd = [
          item.args.ffmpeg, '-y', '-hide_banner',
          '-threads', '1',  # PUT
          '-f', 'rawvideo',
          '-s', f'{W}x{H}',
          '-pix_fmt', 'yuv420p10le',
          '-i', input_fname,
          '-vsync', '0',
          '-y',
          '-pix_fmt', 'rgb24', 
          os.path.join(pngpath_before, "frame_%06d.png") ] 
      
        err = utils.start_process_expect_returncode_0(cmd, wait=True)
        assert err==0, "Generating sequence in YUV format failed."
        
        self.reverseRetargeting(pngpath_before, pngpath_after, roi_update_period, org_image_size_x, org_image_size_y, rtg_image_size_x, rtg_image_size_y, retargeting_gops_rois)

        #cmd = f'{item.args.ffmpeg} -y -hide_banner -i {out_frame_fnames} -f rawvideo -pix_fmt yuv420p10le {output_fname}'
        #cmd1 = cmd.split(' ')
        #err = utils.start_process(cmd1, wait=True)
        cmd = [
          item.args.ffmpeg, '-y', '-hide_banner',
          '-threads', '1',  # PUT
          '-i', os.path.join(pngpath_after, 'frame_%06d.png'),
          '-f', 'rawvideo',
          '-pix_fmt', 'yuv420p10le',
          output_fname] 
        err = utils.start_process_expect_returncode_0(cmd, wait=True)                    
        assert err==0, "Generating sequence in YUV format failed."
        
        
        #cmd = [
        #  item.args.ffmpeg, '-y', '-hide_banner',
        #  '-threads', '1',  # PUT
        #  '-i', os.path.join(pngpath_after, 'frame_%06d.png'),
        #  '-f', 'rawvideo',
        #  '-pix_fmt', 'yuv420p10le',
        #  "test.yuv"] 
        #err = utils.start_process_expect_returncode_0(cmd, wait=True)                    
        #assert err==0, "Generating sequence in YUV format failed."
        #while True:
        #  pass
        #exit()
        
        shutil.rmtree(pngpath_before)
        shutil.rmtree(pngpath_after)
        
        #while True:
        #  pass
        
      elif os.path.isfile(input_fname): 
        os.remove(output_fname)      
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)
        enforce_symlink(input_fname, output_fname)
      else:
        assert False, f"Input file {input_fname} is not found"
      
      return     
      
    # else: retargeting is off:
    
    #if item._is_dir_video:
    if os.path.isdir(input_fname): #item._is_dir_video:
      # video data in a directory
      fnames = sorted(glob.glob(os.path.join(input_fname, '*.png')))
      for idx, fname in enumerate(fnames):
        output_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
        if os.path.isfile(output_frame_fname): os.remove(output_frame_fname)
        os.makedirs(os.path.dirname(output_frame_fname), exist_ok=True)
        enforce_symlink(fname, output_frame_fname)
    else:
      # image data or video in yuv format
      if os.path.isfile(output_fname): os.remove(output_fname)
      os.makedirs(os.path.dirname(output_fname), exist_ok=True)
      enforce_symlink(input_fname, output_fname)


