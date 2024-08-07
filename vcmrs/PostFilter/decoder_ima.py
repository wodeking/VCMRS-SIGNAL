# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import glob
import shutil
from types import SimpleNamespace
from vcmrs.InnerCodec.Utils import nnmanager_utils
from vcmrs.Utils.component import Component
from vcmrs.Utils import data_utils
from vcmrs.Utils.io_utils import enforce_symlink 

class IMA(Component): 
  def __init__(self, ctx):
    super().__init__(ctx)

  def process(self, input_fname, output_fname, item, ctx):
    # make output directory
    if os.path.isfile(output_fname): os.remove(output_fname)
    if os.path.isdir(output_fname): shutil.rmtree(output_fname)
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    if item._is_video:
      if item._is_yuv_video:
        # yuv video
        self._apply_inter_machine_adapter_yuv(input_fname, output_fname, item, ctx)
      else:
        # video in frames 
        os.makedirs(output_fname, exist_ok=True)
        self._apply_inter_machine_adapter_dir(input_fname, output_fname, item, ctx)
    else:
      # image encoding, IMA is always off
      enforce_symlink(input_fname, output_fname) 
 
  def _apply_inter_machine_adapter_dir(self, input_fname, output_fname, item, ctx):
    for idx in range(item.video_info.num_frames):
      in_frame_fname = os.path.join(input_fname, f"frame_{idx:06d}.png")
      out_frame_fname = os.path.join(output_fname, f"frame_{idx:06d}.png")
      if idx in item.video_info.intra_indices and not item.intra_fallback:
        # LIC encoded intra frame, no IMA applied
        # just copy the file intra data
        enforce_symlink(in_frame_fname, out_frame_fname) 
  
      else:
        # prepare parameters
        param = SimpleNamespace()
        param.qp = item.video_info.frame_qps[idx]
        param.intra_fallback = item.intra_fallback # specify if fallback IMA is used
        # prepare video info
        ima_video_info = SimpleNamespace()
        ima_video_info.resolution = item.video_info.resolution #HWC
        #ima_video_info.resolution = (item.video_info.params.height, item.video_info.params.width, 3) #HWC
        ima_video_info.bit_depth = 10
        ima_video_info.chroma_format='420'
        ima_video_info.color_space='yuv'

        # convert frame from png to YUV. Note this is not optimal since IMA converts YUV back 
        # perform ima
        nnmanager_utils.nnmanager_inter_machine_adapter(
            input_fname = in_frame_fname,
            output_fname = out_frame_fname,
            gt_fname = None,
            param = param, 
            video_info = ima_video_info, 
            ctx = ctx)

  def _apply_inter_machine_adapter_yuv(self, input_fname, output_fname, item, ctx):
    # output file
    of = open(output_fname, 'wb')
    for idx in range(item.video_info.num_frames):
      frame_qp = item.video_info.frame_qps[idx]
      #if idx in item.video_info.intra_indices and not item.intra_fallback:
      # QP: -12 indicate an LIC decoded picture. 
      if frame_qp < 0:
        # LIC encoded intra frame, no IMA applied
        # just copy the file intra data
        frame_data = data_utils.get_frame_from_raw_video(input_fname, idx, 
          W=item.video_info.resolution[1],
          H=item.video_info.resolution[0],
          chroma_format='420',
          bit_depth=10,
          return_raw_data=True)
        of.write(frame_data)
  
      else:
        # prepare parameters
        param = SimpleNamespace()
        param.qp = item.video_info.frame_qps[idx]
        param.intra_fallback = item.intra_fallback # specify if fallback IMA is used
        # prepare video info
        ima_video_info = SimpleNamespace()
        ima_video_info.resolution = item.video_info.resolution #HWC
        #ima_video_info.resolution = (item.video_info.params.height, item.video_info.params.width, 3) #HWC
        ima_video_info.bit_depth = 10
        ima_video_info.chroma_format='420'
        ima_video_info.color_space='yuv'

        # perform ima
        ima_out_fname = os.path.join(item.working_dir, 'ima_pf', f"ima_pf_out_{idx}.png")
        ima_out_yuv_fname = os.path.join(item.working_dir, 'ima_pf', f"ima_pf_out_{idx}.yuv")
        os.makedirs(os.path.dirname(ima_out_fname), exist_ok=True)
        nnmanager_utils.nnmanager_inter_machine_adapter(
            input_fname = input_fname+f':{idx}', 
            output_fname = ima_out_fname,
            gt_fname = None,
            param = param, 
            video_info = ima_video_info, 
            ctx = ctx)

        # convert png to YUV
        data_utils.png_to_yuv420p_ffmpeg([ima_out_fname], ima_out_yuv_fname, bitdepth=10, item=item)
        with open(ima_out_yuv_fname, 'rb') as f:
          of.write(f.read())

    of.close()

