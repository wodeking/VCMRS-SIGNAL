# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
import os
import re
import copy
import math
import glob
import math
import asyncio
import shutil
import time
from types import SimpleNamespace

from . import bitstream_utils
from . import nnmanager_utils
from . import video_utils

import vcmrs
from vcmrs.Utils import data_utils
from vcmrs.Utils import utils
from vcmrs.Utils.io_utils import enforce_symlink 
from vcmrs.SpatialResample import adaptivedownsample_data

async def process_input(items, ctx):
  r'''
  Process a file
  '''
  # create semaphore for workers 
  ctx.inner_ctx.worker_sem = asyncio.Semaphore(ctx.input_args.num_workers)
  ctx.inner_ctx.nnmanager_lock = asyncio.Lock() # 1 request to NN Manager

  tasks = []
  for item in items:
    if item._is_video or isIntraFallbackMode(item): # video or image fallback mode
      tasks.append(asyncio.create_task(process_video(item, ctx)))
    else: # image
      tasks.append(asyncio.create_task(process_image(item, ctx)))

  await asyncio.gather(*tasks, return_exceptions=False)



async def process_image(item, ctx):
  '''Process an image
  '''
  fname = item.inner_in_fname

  init_enc_data(item, ctx)

  lic_bitstream_fname = get_working_fname('temp_image_bitstream_fname', item, ctx)
  out_recon_fname = item.inner_out_fname

  # intra config for images 
  video_info = SimpleNamespace()
  intra_cfg = SimpleNamespace()
  intra_cfg.video_qp = item.args.quality 
  intra_cfg.IHA_flag = 0 # Disable intra human adapter 
  intra_cfg.item_tag = f"[{os.path.basename(item.args.working_dir)}]"# Extra information, useful for e.g., logging

  reply = await nnmanager_utils.nnmanager_encode_async(fname, 
    lic_bitstream_fname, 
    ctx = ctx,
    recon_fname = out_recon_fname, 
    video_info=video_info, 
    intra_cfg=intra_cfg)
  assert reply.error==0, 'NNManager encoding failed'

  inner_bitstream_fname = get_inner_output_bitstream_fname(item.working_dir)
  param = SimpleNamespace()
  param.model_id = reply.model_id
  bitstream_utils.gen_image_bitstream(
    input_fname = fname, 
    output_fname = inner_bitstream_fname,
    lic_bitstream_fname = lic_bitstream_fname,
    param = param)

  # generate vcm bitstream
  bitstream_utils.gen_vcm_bitstream(inner_bitstream_fname, item, item.bitstream_fname)

  #utils.exit_enc_data(cfg)

async def process_video(item, ctx):
  r'''
  Process a video
  ''' 
  fname = item.inner_in_fname
  vcmrs.log('processing video ...')

  init_enc_data(item, ctx)

  chunk_info_list = []

  # process video in frames format
  video_info = video_utils.get_video_info(fname, item.args)
  item.video_info = video_info
  
  frame_fnames = video_info.frame_fnames
  num_frames = video_info.num_frames - item.args.FrameSkip
  if item.args.FramesToBeEncoded > 0: num_frames = min(num_frames, item.args.FramesToBeEncoded)

  # get frame resolution
  H,W,C = video_info.resolution

  # prepare video working dir
  video_working_dir = get_working_fname('video_working_dir', item, ctx)

  # preprocess the video, for example, padding if needed
  preprocess_video(video_info, video_working_dir, item)

  # prepare the whole video sequence for CVC in intra fallback mode
  if isIntraFallbackMode(item):
    prepare_cvc_input_fallback(video_info, video_working_dir, item)
  else: # NNVVC inner codec
    # inner_config = {"config": item.args.Configuration}
    # param_data = json.dumps(inner_config).encode()
    # param_data = bytearray(param_data)
    param_data = bytearray(item.args.Configuration.encode())
    item.add_parameter('InnerCodec', param_data=param_data) 
  
  # process frames in chunks 
  # divide video into chunks, each chunk is a RA segment
  if item.args.single_chunk:
    item.chunk_size = num_frames
  else:
    item.chunk_size = item.IntraPeriod

  video_start_idx = item.args.FrameSkip
  video_end_idx = video_start_idx + num_frames - 1 # inclusive

  # prepare chunk information
  n_chunks = math.ceil(num_frames / item.chunk_size)
  for chunk_idx in range(n_chunks):
    chunk_info = SimpleNamespace()
    chunk_info.working_dir = video_working_dir
    chunk_info.video_info = video_info
    chunk_info.chunk_idx = chunk_idx
    chunk_info.start_idx = video_start_idx + chunk_idx * item.chunk_size
    chunk_info.end_idx = min(chunk_info.start_idx + item.chunk_size, video_end_idx)
    chunk_info.intra_period = item.IntraPeriod
    chunk_info_list.append(chunk_info)

  # process chunks in parallel
  if ctx.input_args.debug and (ctx.input_args.num_workers == 1):
    # debug
    for chunk_info in chunk_info_list:
      await process_chunk(chunk_info, item, ctx)
  else:
    tasks = []
    for chunk_info in chunk_info_list:
      task = asyncio.create_task(process_chunk(chunk_info, item, ctx))
      tasks.append(task)


    # wait until all chunks are handled
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # cancel unfinished tasks
    for task in tasks: 
      task.cancel() 
    
  # collect bitstream 
  # get bitstream name
  bname = os.path.basename(fname)
  # for video in png files, the fname is a folder
  if not bname: bname = os.path.basename(os.path.dirname(fname))
  # still empty
  if not bname: bname = 'bitstream'
  out_bitstream_fname = item.bitstream_fname #os.path.join(item.bitstream_dir, bname+'.bin')

  if item.args.VCMBitStructOn:
    rsd_fname = get_rsd_fname(video_working_dir)
    bitstream_utils.write_restoration_data(rsd_fname, item)

  # inner codec bitstream name
  if not item.args.VCMBitStructOn:
    inner_bitstream_fname = get_inner_output_bitstream_fname(video_working_dir)
  else:
    inner_bitstream_fname = out_bitstream_fname
  gen_inner_bitstream(chunk_info_list, inner_bitstream_fname, video_working_dir, item)

  # generate vcm bitstream
  if not item.args.VCMBitStructOn:
    bitstream_utils.gen_vcm_bitstream(inner_bitstream_fname, item, out_bitstream_fname)
  
  
  # VCM recon file generation
  gen_vcm_recon(chunk_info_list, item, ctx, video_working_dir)

  # set extra information to item 
  #if ZJU_VCM_SINGLE
  set_extra_info_item(item, ctx, video_working_dir, chunk_info_list)
  #else
  # set_extra_info_item(item, ctx, video_working_dir)
  #endif
 
  #utils.exit_enc_data(cfg)

async def process_chunk(chunk_info, item, ctx):
  '''
  Process a chunk of a video. A chunk is a number of segments, except the last chunk of a video. 
  The minimal size of a chunk is a segment. 
  The intra frames are first processed. 

  Args:
    chunk_info:
      .chunk_idx
      .chunk_start_idx
      .chunk_end_idx
      .intra_period
      .video_info
  '''
  vcmrs.log(f'processing chunk {chunk_info.chunk_idx}: {chunk_info.start_idx} - {chunk_info.end_idx}')
  ###########################################
  # fallback mode
  if isIntraFallbackMode(item):
    await process_chunk_fallback(chunk_info, item, ctx)
    return

  ###########################################
  # process intra frames

  # video info passing to NNManager
  if video_utils.is_yuv_video(chunk_info.video_info):
    pass_video_info = chunk_info.video_info 
  else:
    pass_video_info = None 

  intra_working_dir = get_chunk_intra_working_dir(chunk_info)
  os.makedirs(intra_working_dir, exist_ok=True)

  intra_info = get_chunk_intra_info(chunk_info)
  for intra_data in intra_info:
      # video info is only used for YUV video file
      await process_intra(
        intra_data,
        item,
        ctx,
        video_info=pass_video_info)

  vcmrs.log('Intra data has been processed')

  ###########################################
  # combine intra with inter frame and generate YUV video sequence
  cvc_intra_fnames = [x.cvc_intra_fname for x in intra_info]
  chunk_mix_intra_inter_fname = get_chunk_mix_intra_inter_fname(chunk_info)
  #if ZJU_VCM_SINGLE
  chunk_nnic_dec_iha_fname = get_chunk_all_intra_fname(chunk_info)
  merge_intra_inter_frames(cvc_intra_fnames, chunk_info, chunk_mix_intra_inter_fname, chunk_nnic_dec_iha_fname, item)
  #else
  # merge_intra_inter_frames(cvc_intra_fnames, chunk_info, chunk_mix_intra_inter_fname, item)
  #endif
  vcmrs.log("Merged intra and inter frames.")

  ###########################################
  # Call CVC Intra Encoder to encode the sequence 
  mix_bitstream_fname = get_chunk_cvc_output_bs_fname(chunk_info)
  mix_recon_fname = get_chunk_cvc_output_recon_fname(chunk_info) 
  mix_cvc_log_fname = get_chunk_cvc_log_fname(chunk_info) if ctx.input_args.debug else None 


  await encode_mixed_intra_inter(chunk_mix_intra_inter_fname, 
    mix_bitstream_fname,
    mix_recon_fname,
    chunk_info,
    mix_cvc_log_fname, 
    item,
    ctx)

  if (not ctx.input_args.debug) and os.path.isfile(chunk_mix_intra_inter_fname):
    os.remove(chunk_mix_intra_inter_fname)

  #############################################
  # inter machine adapter
  if item.args.InterMachineAdapter or not item._is_yuv_video:
    apply_inter_machine_adapter(chunk_info, item, ctx)

  return True


async def process_intra(
    intra_data,
    item,
    ctx, 
    video_info=None):
  '''
  Process an intra frame

  Return:
    bitstream_fname and recon_fname 
  '''

  # determin intra cfg
  intra_cfg = SimpleNamespace()
  intra_cfg.video_qp = item.args.quality 
  intra_cfg.IHA_flag = 0 # IHA is skipped for image coding
  if item._is_video:
    intra_cfg.item_tag = f"[{os.path.basename(item.args.working_dir)}]"# Extra information, useful for e.g., logging
    # video coding, apply NNIntraQPOffset
    intra_cfg.video_qp += item.args.NNIntraQPOffset
    intra_cfg.IHA_flag = item.args.IntraHumanAdapter # apply intra human adapter, default true

  cvc_intra_fname = intra_data.cvc_intra_fname

  reply = await nnmanager_utils.nnmanager_encode_async(
    input_fname = intra_data.intra_fname, 
    bitstream_fname = intra_data.out_bitstream_fname, 
    recon_fname = intra_data.out_recon_fname, 
    param_fname = intra_data.out_param_fname,
    intra_cfg = intra_cfg,
    cvc_intra_fname = cvc_intra_fname,
    video_info = video_info,
    ctx = ctx)

  assert reply.error==0, f'intra processing failed with error code: {reply.error}'

  # write some dummy parameters to config file
  intra_param = SimpleNamespace()
  intra_param.model_id = reply.model_id
  intra_param.IHA_flag =  intra_cfg.IHA_flag # todo: intra_human_adapter usage shall be reported by NNManager
  intra_param.IHA_patch = 0 # todo: this flag should come from NNManager 
  bitstream_utils.write_params(intra_data.out_param_fname, intra_param)
 
async def encode_mixed_intra_inter(in_fname, out_bitstream_fname, out_recon_fname, chunk_info, cvc_log_fname, item, ctx):
  '''
  Encode the mixed intra-inter sequence
  '''

  cvc_dir = ctx.input_args.cvc_dir
  cvc_encoder=os.path.join(cvc_dir, 'bin', 'EncoderAppStatic')
  intra_period = chunk_info.intra_period
  alf_flag = ""
  if item.args.Configuration == 'RandomAccess':
    if item.IntraPeriod<16:
      cvc_encoder_cfg1 = os.path.join(cvc_dir, 'cfg', 'encoder_randomaccess_vtm_gop8.cfg')
    elif item.IntraPeriod<32:
      cvc_encoder_cfg1 = os.path.join(cvc_dir, 'cfg', 'encoder_randomaccess_vtm_gop16.cfg')
    else:
      cvc_encoder_cfg1=os.path.join(cvc_dir, 'cfg', 'encoder_randomaccess_vtm.cfg')
    alf_flag = "--ALFDisableInNonRef"
  elif item.args.Configuration == 'LowDelay':
    cvc_encoder_cfg1=os.path.join(cvc_dir, 'cfg', 'encoder_lowdelay_vtm.cfg')
    intra_period = -1
  elif item.args.Configuration == 'AllIntra':
    cvc_encoder_cfg1=os.path.join(cvc_dir, 'cfg', 'encoder_intra_vtm_no_subsample.cfg')
    intra_period = 1
  else:
    cvc_encoder_cfg1=item.args.Configuration

  cvc_encoder_cfg2=os.path.join(cvc_dir, 'cfg', 'lossless', 'lossless_intra.cfg')
  H,W,C = chunk_info.video_info.resolution

  cmd = [
    cvc_encoder, 
    '-c', cvc_encoder_cfg1,
    #if ZJU_VCM_SINGLE
    # '-c', cvc_encoder_cfg2,
    #endif
    f'--InputFile={in_fname}', 
    f'--SourceWidth={W}',
    f'--SourceHeight={H}',
    '--InputBitDepth=10',
    f'--BitstreamFile={out_bitstream_fname}',
    f'--ReconFile={out_recon_fname}',
    f'--FrameRate={chunk_info.video_info.frame_rate}',
    f'--IntraPeriod={intra_period}',
    f'--FramesToBeEncoded={chunk_info.end_idx - chunk_info.start_idx + 1}',
    f'--QP={item.args.quality}',
    #f'--ConformanceWindowMode=1',
    alf_flag,
    #if ZJU_VCM_SINGLE
    '--SingleLayerVCM=1',
    #endif
  ]

  # acquire worker resources
  await ctx.inner_ctx.worker_sem.acquire()
  try: 
    vcmrs.log('CVC encoding...')
    # vcmrs.debug(cmd)
    # Todo: fast debug disabled
    if ctx.input_args.debug and ctx.input_args.debug_skip_vtm:
      err = 0 
    else:
      err = await utils.start_process_async(cmd, log_fname=cvc_log_fname, time_tag=f"[{os.path.basename(item.args.working_dir)}] CVC encoding done")

  finally:
    ctx.inner_ctx.worker_sem.release()

  assert err==0, 'CVC Encoding failed: ' + f'{" ".join(cmd)}'
  vcmrs.log('=======================================')

def apply_inter_machine_adapter(chunk_info, item, ctx):

  # loop all inter frames and apply inter machine adapter
  cvc_output_video_fname = get_chunk_cvc_output_recon_fname(chunk_info)

  # get frame QP value from the bitstream 
  cvc_output_bs_fname = get_chunk_cvc_output_bs_fname(chunk_info)
  #if ZJU_VCM_SINGLE
  if isIntraFallbackMode(item):
    chunk_all_intra_fname = ""
  else:
    chunk_all_intra_fname = get_chunk_all_intra_fname(chunk_info)
  frame_info_dict = get_frame_info_from_cvc_bitstream(cvc_output_bs_fname, ctx, chunk_all_intra_fname)
  #else
  # frame_info_dict = get_frame_info_from_cvc_bitstream(cvc_output_bs_fname, ctx)
  #endif

  # video info to be passed to NNManager for inter machine adapter
  ima_video_info = SimpleNamespace()
  ima_video_info.bit_depth = 10
  ima_video_info.chroma_format = '420'
  ima_video_info.color_space = 'yuv'
  ima_video_info.resolution = chunk_info.video_info.resolution
 

  chunk_ima_output_dir = get_chunk_ima_output_dir(chunk_info)
  os.makedirs(chunk_ima_output_dir, exist_ok=True)
  for idx, gt_fname in enumerate(get_chunk_frame_interator(chunk_info)):
    if idx % chunk_info.intra_period != 0 or isIntraFallbackMode(item):
      # skip intra frames if not in fallback mode
      cvc_output_frame_fname = cvc_output_video_fname + f":{idx}"
      ima_output_frame_fname = get_chunk_ima_output_recon_fname(chunk_info, idx)

      if item.args.InterMachineAdapter:
        # apply inter machine adapter
        param = SimpleNamespace()
        param.item_tag = f"[{os.path.basename(item.args.working_dir)}]"# Extra information, useful for e.g., logging
        param.qp = frame_info_dict[idx].qp
        param.intra_fallback = isIntraFallbackMode(item)
        nnmanager_utils.nnmanager_inter_machine_adapter(
          input_fname = cvc_output_frame_fname, 
          output_fname = ima_output_frame_fname,
          gt_fname = gt_fname, 
          param = param, 
          video_info = ima_video_info, 
          ctx = ctx)
      else:
        # perform color conversion
        if item.args.InnerCodec == 'VTM' and (not item._is_yuv_video):
          data_utils.yuv420p10b_to_png_ffmpeg(
            cvc_output_frame_fname, 
            ima_video_info.resolution, 
            ima_output_frame_fname,
            item)
        else:
          data_utils.yuv420p10b_to_png(
            cvc_output_frame_fname, 
            ima_video_info.resolution,
            ima_output_frame_fname)


def gen_inner_bitstream(chunk_info_list, out_bitstream_fname, video_working_dir, item):
  '''
  Generate VCM bitstream from bitstreams of chunks
  '''

  # merge cvc output bitstreams of the chunks
  video_cvc_output_bs_fname = get_video_cvc_output_bs_fname(video_working_dir)

  bs_fnames = []
  for chunk_info in chunk_info_list:
    bs_fnames.append(get_chunk_cvc_output_bs_fname(chunk_info))

  if len(bs_fnames)>1:
    cvc_bs_merger=os.path.join(item.args.cvc_dir, 'bin', 'parcatStatic')

    cmd = [ cvc_bs_merger ] 
    cmd += bs_fnames
    cmd.append(video_cvc_output_bs_fname)
    
    vcmrs.debug(cmd)
    err = utils.start_process(cmd, wait=True)
    assert err==0, 'merging traunk cvc output bitstream failed'
  else:
    shutil.copy(bs_fnames[0], video_cvc_output_bs_fname)

  # call vcm mux to merge the intra bitstreams with cvc bitstream
  vcm_bs_mux = os.path.join(item.args.cvc_dir, 'bin', 'FrameMixerEncAppStatic')

  # in intra fallback mode, directly return the concatenated bitstream file
  if isIntraFallbackMode(item):
    #shutil.copy(video_cvc_output_bs_fname, out_bitstream_fname)

    # using IMA_model_id to transfer ima on/off and padding information
    # ima status off << 16
    ima_param = SimpleNamespace()
    ima_param.IMA_model_id = chunk_info_list[0].video_info.pad_h*8 + \
      chunk_info_list[0].video_info.pad_w + \
      (not item.args.InterMachineAdapter) * 64
    ima_param_data_dir = get_ima_param_data_dir(video_working_dir) 
    os.makedirs(ima_param_data_dir, exist_ok=True)
    ima_param_fname = get_ima_param_fname(video_working_dir, 0)
    bitstream_utils.write_params(ima_param_fname, ima_param)

    ima_param_fname_base = get_ima_param_fname_base(video_working_dir)
    if item.args.VCMBitStructOn:
      rsd_fname = get_rsd_fname(video_working_dir)

    cmd = [vcm_bs_mux]
    cmd += ['-b', video_cvc_output_bs_fname] # cvc output bitstream
    cmd += ['-o', out_bitstream_fname] # output
    cmd += ['-s', ima_param_fname_base] # ima parameter basename
    cmd += ['-p', 1] # remove duplicate parameter set
    if item.args.VCMBitStructOn:
      vcmrs.log('VCM bistream strucutre is on at encoder stage 1.\n')
      cmd += ['-r', rsd_fname] # restoration data file name

 
    vcmrs.debug(cmd)
    err = utils.start_process(cmd, wait=True)
    assert err==0, 'vcm mux failed at intra fallback mode'

    return

  # prepare intra bitstreams
  intra_bs_fnames = []
  intra_param_fnames = []
  for chunk_idx,chunk_info in enumerate(chunk_info_list):
    intra_info = get_chunk_intra_info(chunk_info)
    for intra_idx, intra_data in enumerate(intra_info):
      if chunk_idx==0 or intra_idx!=0:
        intra_bs_fnames.append(intra_data.out_bitstream_fname)
        intra_param_fnames.append(intra_data.out_param_fname)

  # copy all intra bitstream and param into one folder
  video_intra_data_dir = get_video_intra_data_dir(video_working_dir) 
  os.makedirs(video_intra_data_dir, exist_ok=True)
  bs_base = os.path.join(video_intra_data_dir, 'intra_bs_')
  param_base = os.path.join(video_intra_data_dir, 'intra_param_')
  for idx, (bs_fname, param_fname) in enumerate(zip(intra_bs_fnames, intra_param_fnames)):
    # rename to handle if the link already exists
    temp_link = os.path.join(video_working_dir, '.temp_link')
    enforce_symlink(os.path.realpath(bs_fname), temp_link) 
    os.rename(temp_link, bs_base+f'{idx:06d}.bin')
    enforce_symlink(os.path.realpath(param_fname), temp_link) 
    os.rename(temp_link, param_base+f'{idx:06d}.txt')

  ima_param_data_dir = get_ima_param_data_dir(video_working_dir) 
  os.makedirs(ima_param_data_dir, exist_ok=True)
  # set scale factor to the first intra picture
  # set padding info to the first intra picture
  ima_param = SimpleNamespace()
  # using IMA_model_id to transfer padding
  ima_param.IMA_model_id = chunk_info_list[0].video_info.pad_h*8 + \
    chunk_info_list[0].video_info.pad_w + \
      (not item.args.InterMachineAdapter) * 64

  #  #ima_param.IMA_model_id = scale_factor_dict[chunk_info_list[0].video_info.scale_factor]
  ima_param_fname = get_ima_param_fname(video_working_dir, 0)
  bitstream_utils.write_params(ima_param_fname, ima_param)

  # collect ima parameters
  for chunk_idx,chunk_info in enumerate(chunk_info_list):
    # note finding files in a dictory has a problem that when the working directory is dirty,
    # it may read wrong files
    chunk_ima_output_dir = get_chunk_ima_output_dir(chunk_info)
    for chunk_ima_param_fname in glob.glob(os.path.join(chunk_ima_output_dir, 'adapter_param_*.txt')):
      chunk_frame_idx = int(os.path.splitext(os.path.basename(chunk_ima_param_fname))[0].split('_')[-1])
      video_frame_idx = chunk_idx*chunk_info_list[0].intra_period + chunk_frame_idx
      assert video_frame_idx != 0, 'The first intra frame is used to transfer the IMA status and padding information. This logic may be changed later.'
      video_ima_param_fname = get_ima_param_fname(video_working_dir, video_frame_idx)
      # chunk_ima_param_fname ==> video_ima_param_fname
      temp_link = os.path.join(video_working_dir, '.temp_link')
      enforce_symlink(os.path.realpath(chunk_ima_param_fname), temp_link) 
      os.rename(temp_link, video_ima_param_fname)

  ima_param_fname_base = get_ima_param_fname_base(video_working_dir)

  if item.args.VCMBitStructOn:
    rsd_fname = get_rsd_fname(video_working_dir)

  cmd = [vcm_bs_mux]
  cmd += ['-b', video_cvc_output_bs_fname] # cvc output bitstream
  cmd += ['-o', out_bitstream_fname] # output
  cmd += ['-i', bs_base] # intra bitstream basename
  cmd += ['-c', param_base] # intra paramter basename
  cmd += ['-s', ima_param_fname_base] # ima parameter basename
  cmd += ['-p', 1] # remove duplicate parameter set
  if item.args.VCMBitStructOn:
    vcmrs.log('VCM bistream strucutre is on at encoder stage 2.\n')
    cmd += ['-r', rsd_fname] # restoration data file name

  vcmrs.debug(cmd)
  err = utils.start_process(cmd, wait=True)
  assert err==0, 'vcm mux failed'


def gen_vcm_recon(chunk_info_list, item, ctx, video_working_dir):
  '''
  Generate reconstructed frames of a chunk
  '''
  if item._is_video:
    # YUV as input and no IMA
    if item._is_yuv_video and (not item.args.InterMachineAdapter):
      if isIntraFallbackMode(item):
        # concatenate chunk YUVs
        yuv_fnames = []
        for chunk_info in chunk_info_list:
          mix_recon_fname = get_chunk_cvc_output_recon_fname(chunk_info)
          yuv_fnames.append(mix_recon_fname)
          H,W,C = chunk_info.video_info.resolution
        data_utils.concat_yuvs(
          yuv_fnames, 
          width=W, 
          height=H, 
          bitdepth=10, 
          num_overlap_frames=1, 
          out_fname=item.inner_out_fname)  
        return
 
      recon_dir = get_inner_output_png_dir(video_working_dir)
      yuv_fnames = []
      for chunk_info in chunk_info_list:
        for idx in range(chunk_info.end_idx - chunk_info.start_idx+1):
          is_intra = (idx % chunk_info.intra_period) == 0
          if is_intra:
            # png file
            intra_working_dir = get_chunk_intra_working_dir(chunk_info)
            intra_idx = idx // chunk_info.intra_period
            chunk_frame_fname = get_chunk_intra_recon_fname(intra_working_dir, intra_idx)

            # convert png output to YUV
            frame_idx = chunk_info.start_idx + idx
            in_frame_bname = f'frame_{frame_idx:06d}'
            out_frame_fname = os.path.join(recon_dir, in_frame_bname+'.png')
            if os.path.exists(out_frame_fname):
              continue

            # unpad if necessary 
            unpad_frame(chunk_frame_fname, out_frame_fname, 
              chunk_info.video_info.resolution,
              chunk_info.video_info.pad_h,
              chunk_info.video_info.pad_w,
              ffmpeg=item.args.InnerCodec=='VTM',
              item=item)

            # convert png to yuv 
            out_yuv_frame_fname = os.path.join(recon_dir, in_frame_bname+'.yuv')
            data_utils.png_to_yuv420p_ffmpeg([out_frame_fname], out_yuv_frame_fname, bitdepth=10, item=item)
            if not (idx==0 and len(yuv_fnames)>0):
              yuv_fnames.append(out_yuv_frame_fname)

          else:
            # get cvc YUV output
            out_yuv_frame_fname= get_chunk_cvc_output_recon_fname(chunk_info) + f':{idx}'

            yuv_fnames.append(out_yuv_frame_fname)

      # concat all yuv files
      H,W,C = chunk_info.video_info.resolution
      data_utils.concat_yuvs(yuv_fnames, width=W, height=H, bitdepth=10, num_overlap_frames=0, out_fname=item.inner_out_fname)  
      return

    # video processing
    if item._is_yuv_video:
      recon_dir = get_inner_output_png_dir(video_working_dir)
    else:
      recon_dir = item.inner_out_fname

    for chunk_info in chunk_info_list:
      # move converted png file to recon_dir
      for idx in range(chunk_info.end_idx - chunk_info.start_idx+1):
        is_intra = (idx % chunk_info.intra_period) == 0

        if is_intra and not isIntraFallbackMode(item):
          intra_working_dir = get_chunk_intra_working_dir(chunk_info)
          intra_idx = idx // chunk_info.intra_period
          chunk_frame_fname = get_chunk_intra_recon_fname(intra_working_dir, intra_idx)
        else:
          chunk_frame_fname = get_chunk_ima_output_recon_fname(chunk_info, idx)

        frame_idx = chunk_info.start_idx + idx

        if video_utils.is_yuv_video(chunk_info.video_info):
          in_frame_bname = f'frame_{frame_idx:06d}'
        else:
          in_frame_fname = chunk_info.video_info.frame_fnames[idx + chunk_info.start_idx]
          in_frame_bname = os.path.splitext(os.path.basename(in_frame_fname))[0]
        out_frame_fname = os.path.join(recon_dir, in_frame_bname+'.png')
        if os.path.exists(out_frame_fname):
          continue

        # unpad if necessary 
        unpad_frame(chunk_frame_fname, out_frame_fname, 
          chunk_info.video_info.resolution,
          chunk_info.video_info.pad_h,
          chunk_info.video_info.pad_w,
          ffmpeg=item.args.InnerCodec=='VTM',
          item=item)

    # convert png to yuv if input is yuv format
    if item._is_yuv_video:
      out_frame_fnames = sorted(glob.glob(os.path.join(recon_dir, '*.png')))
      data_utils.png_to_yuv420p_ffmpeg(out_frame_fnames, item.inner_out_fname, bitdepth=10, item=item)

  else: 
    # image processing
    assert len(chunk_info_list)==1, "Internal error: The chunk_info_list shall contain one item for image processing"

    chunk_info = chunk_info_list[0]
    # move converted png file to recon_dir
    idx = 0

    if not isIntraFallbackMode(item):
      intra_working_dir = get_chunk_intra_working_dir(chunk_info)
      intra_idx = 0
      chunk_frame_fname = get_chunk_intra_recon_fname(intra_working_dir, intra_idx)
    else:
      chunk_frame_fname = get_chunk_ima_output_recon_fname(chunk_info, idx)

    out_frame_fname = item.inner_out_fname

    # unpad if necessary
    unpad_frame(chunk_frame_fname, out_frame_fname, 
          chunk_info.video_info.resolution,
          chunk_info.video_info.pad_h,
          chunk_info.video_info.pad_w,
          ffmpeg=item.args.InnerCodec=='VTM',
          item=item)

 
def unpad_frame(in_fname, out_fname, resolution, pad_h, pad_w, ffmpeg, item):
  # unpad if necessary
  # create output directory if needed
  os.makedirs(os.path.dirname(out_fname), exist_ok=True)
  if (pad_h==0) and (pad_w==0):
    shutil.copy(in_fname, out_fname)
  else:
    H,W,C = resolution
    if ffmpeg:
      data_utils.unpad_image_ffmpeg(
        in_fname,
        out_fname, 
        W-pad_w, 
        H-pad_h, 
        item) 
    else:
      data_utils.unpad_image(in_fname, 
        out_fname, 
        pad_h, 
        pad_w) 


########################################################
# fallback mode
def prepare_cvc_input_fallback(video_info, video_working_dir, item):
  # convert input to YUV420 10-bit
  H,W,C = video_info.resolution
  input_yuv_fname = os.path.join(video_working_dir, f"fallback_cvc_input_{W}x{H}.yuv")
  video_info.fallback_input_yuv_fname  = input_yuv_fname

  if item.args.InnerCodec == 'VTM':
    # When VTM is used as inner codec, the prepocessing is slightly different. 
    # the input PNG data are converted into YUV 420 8bit using ffmpeg. 
    if item._is_yuv_video:
      # copy video_info.frame_fnames to input_yuv_fname
      if os.path.isfile(input_yuv_fname): os.remove(input_yuv_fname)
      enforce_symlink(os.path.realpath(video_info.frame_fnames), input_yuv_fname) 

    else:
      input_frames = video_info.frame_fnames
      data_utils.png_to_yuv420p_ffmpeg(input_frames, input_yuv_fname, bitdepth=8, item=item)
      
    return

  if video_utils.is_yuv_video(video_info):
    cmd = [
       item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
       '-threads', '1',
       '-f', 'rawvideo', 
       '-pix_fmt', video_utils.get_ffmpeg_pix_fmt(video_info),
       '-s', f'{W}x{H}',
       '-i', video_info.frame_fnames,
       '-f', 'rawvideo', 
       '-pix_fmt', 'yuv420p10le', 
       input_yuv_fname]

    err = utils.start_process_expect_returncode_0(cmd, wait=True)
    assert err==0, 'Generating sequence in YUV format failed'
 
  else:
    # directory of png images
    # color format conversion
    input_frames = video_info.frame_fnames
    data_utils.png_to_yuv420p10b(input_frames, input_yuv_fname)


async def process_chunk_fallback(chunk_info, item, ctx):
  # Call CVC Intra Encoder to encode the sequence 
  mix_bitstream_fname = get_chunk_cvc_output_bs_fname(chunk_info)
  mix_recon_fname = get_chunk_cvc_output_recon_fname(chunk_info) 
  mix_cvc_log_fname = get_chunk_cvc_log_fname(chunk_info) if item.args.debug else None 

  cvc_encoder=os.path.join(ctx.input_args.cvc_dir, 'bin', 'EncoderAppStatic')
  numFrames = chunk_info.end_idx - chunk_info.start_idx + 1
  intra_period = chunk_info.intra_period
  alf_flag = ""

  if item._is_video:
    # video coding
    if item.args.Configuration == 'RandomAccess':
      if item.IntraPeriod < 16:
        cvc_encoder_cfg1 = os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_randomaccess_vtm_gop8.cfg')
      elif item.IntraPeriod < 32:
        cvc_encoder_cfg1 = os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_randomaccess_vtm_gop16.cfg')
      else:
        cvc_encoder_cfg1 = os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_randomaccess_vtm.cfg')
      alf_flag = "--ALFDisableInNonRef"
    elif item.args.Configuration == 'LowDelay':
      cvc_encoder_cfg1=os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_lowdelay_vtm.cfg')
      intra_period = -1
    elif item.args.Configuration == 'AllIntra':
      cvc_encoder_cfg1=os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_intra_vtm_no_subsample.cfg')
      intra_period = 1
    else:
      # user specified configuration file
      cvc_encoder_cfg1 = item.args.Configuration

  else:
    # image coding
    cvc_encoder_cfg1=os.path.join(ctx.input_args.cvc_dir, 'cfg', 'encoder_intra_vtm.cfg')
    intra_period = 1
  H,W,C = chunk_info.video_info.resolution

  if item.args.InnerCodec == 'VTM':
    # When VTM is used as inner codec, the input YUV data is in 8 bit
    cmd = [
      cvc_encoder, 
      '-c', cvc_encoder_cfg1,
      f'--InputFile={chunk_info.video_info.fallback_input_yuv_fname}', 
      f'--SourceWidth={W}',
      f'--SourceHeight={H}',
      f'--InputBitDepth={chunk_info.video_info.bit_depth}',
      f'--BitstreamFile={mix_bitstream_fname}',
      f'--ReconFile={mix_recon_fname}',
      f'--FrameRate={chunk_info.video_info.frame_rate}',
      f'--IntraPeriod={intra_period}',
      f'--FrameSkip={chunk_info.start_idx}',
      f'--FramesToBeEncoded={numFrames}',
      f'--QP={item.args.quality}',
      '--InternalBitDepth=10',
      '--ConformanceWindowMode=1',
    ]
    if item.args.SpatialResample == 'AdaptiveDownsample':
      scale_list_path=os.path.join(item.args.working_dir, "scalelist", f"{item.args.working_dir.split(os.path.sep)[-1]}_{chunk_info.chunk_idx}.txt")
      cmd.extend([ 
                  '-c', f'{os.path.join(ctx.input_args.cvc_dir, "cfg", "rpr","MIRPR.cfg")}',
                  f'--ViscomRPRScaleListFile={scale_list_path}',
                  f'--ScalingRatioHor={1/adaptivedownsample_data.id_scale_factor_mapping[1]:.2f}',
                  f'--ScalingRatioVer={1/adaptivedownsample_data.id_scale_factor_mapping[1]:.2f}',
                  f'--ScalingRatioHor2={1/adaptivedownsample_data.id_scale_factor_mapping[2]:.2f}',
                  f'--ScalingRatioVer2={1/adaptivedownsample_data.id_scale_factor_mapping[2]:.2f}',
                  f'--ScalingRatioHor3={1/adaptivedownsample_data.id_scale_factor_mapping[3]:.2f}',
                  f'--ScalingRatioVer3={1/adaptivedownsample_data.id_scale_factor_mapping[3]:.2f}',
                  f'--ScalingRatioHor4={1/adaptivedownsample_data.id_scale_factor_mapping[4]:.2f}',
                  f'--ScalingRatioVer4={1/adaptivedownsample_data.id_scale_factor_mapping[4]:.2f}',
                  ])
  else:
    cmd = [
      cvc_encoder, 
      '-c', cvc_encoder_cfg1,
      f'--InputFile={chunk_info.video_info.fallback_input_yuv_fname}', 
      f'--SourceWidth={W}',
      f'--SourceHeight={H}',
      '--InputBitDepth=10',
      f'--BitstreamFile={mix_bitstream_fname}',
      f'--ReconFile={mix_recon_fname}',
      f'--FrameRate={chunk_info.video_info.frame_rate}',
      f'--IntraPeriod={intra_period}',
      f'--FrameSkip={chunk_info.start_idx}',
      f'--FramesToBeEncoded={numFrames}',
      f'--QP={item.args.quality}',
      alf_flag,
      #"--ALFDisableInNonRef"
    ]

  # acquire worker resources
  await ctx.inner_ctx.worker_sem.acquire()
  try: 
    vcmrs.log('CVC encoding...')
    vcmrs.log(cmd)

    if ctx.input_args.debug and ctx.input_args.debug_skip_vtm:
      err = 0 
    else:
      err = await utils.start_process_async(cmd, log_fname=mix_cvc_log_fname, time_tag=f"[{os.path.basename(item.args.working_dir)}] CVC encoding done")
    

  finally:
    ctx.inner_ctx.worker_sem.release()

  assert err==0, f'CVC Encoding failed with command `{cmd}`'
  vcmrs.log('=======================================')

  #############################################
  # inter machine adapter
  if item.args.InterMachineAdapter or not item._is_yuv_video:
    apply_inter_machine_adapter(chunk_info, item, ctx)

  return True

########################################################
# intemediate file names

# get temporary file name or directory name
def get_working_fname(tgt, item, ctx):
  bname = os.path.basename(item.inner_in_fname)
  if tgt == 'temp_image_bitstream_fname':
    # the temporary bitstream file generated by intra codec
    return os.path.join(item.inner_working_dir, bname+'.intra.bin') 
  elif tgt == 'image_recon_fname':
    # todo: remove this
    # recon image generated by intra codec
    return os.path.join(item.inner_working_dir, bname+'.png') 
  elif tgt == 'inner_image_bitstream_fname':
    return os.path.join(item.inner_working_dir, 'inner_codec.bin') 
  elif tgt == 'image_bitstream_fname':
    # todo: remove this, bitstream name is determined by the outer encoder
    return os.path.join(item.bitstream_dir, os.path.splitext(item._bname)[0]+'.bin')

  elif tgt == 'video_working_dir':
    video_working_dir = os.path.join(item.working_dir, 'nnvvc_video')
    os.makedirs(video_working_dir, exist_ok=True)
    return video_working_dir



def get_temp_image_bitstream_fname(input_fname, item):
  bname = os.path.basename(input_fname)
  return os.path.join(item.working_dir, bname+'.intra.bin')

def get_image_bitstream_fname(input_fname, cfg):
  bname = os.path.splitext(os.path.basename(input_fname))[0]
  return os.path.join(cfg.bitstream_dir, bname + '.bin')

def get_image_recon_fname(input_fname, cfg):
  bname = os.path.splitext(os.path.basename(input_fname))[0]
  return os.path.join(cfg.recon_dir, bname + '.png')


def get_chunk_intra_working_dir(chunk_info):
  return os.path.join(chunk_info.working_dir, f'chunk_intra_{chunk_info.chunk_idx}')

def get_chunk_intra_recon_fname(intra_working_dir, intra_idx):
  return os.path.join(intra_working_dir, f'intra_recon_{intra_idx:06d}.png')

#if ZJU_VCM_SINGLE
def get_chunk_intra_recon_yuv_fname(intra_working_dir, intra_idx, video_info):
  H,W,_ = video_info.resolution
  return os.path.join(intra_working_dir, f'intra_cvc_{intra_idx:06d}_{W}x{H}.yuv')
#endif

def get_video_intra_data_dir(video_working_dir):
  return os.path.join(video_working_dir, f'video_intra_data')

def get_ima_param_data_dir(video_working_dir):
  return os.path.join(video_working_dir, f'ima_param_data')

def get_ima_param_fname(video_working_dir, frame_idx):
  ima_param_dir = get_ima_param_data_dir(video_working_dir)
  return os.path.join(ima_param_dir, f'adapter_param_{frame_idx:06d}.txt')

def get_ima_param_fname_base(video_working_dir):
  ima_param_dir = get_ima_param_data_dir(video_working_dir)
  return os.path.join(ima_param_dir, f'adapter_param_')

def get_rsd_fname(video_working_dir):
  return os.path.join(video_working_dir, f'tmp_rsd.bin')

# get fnames related to intra frame processing
def get_chunk_intra_info(chunk_info):
  intra_working_dir = get_chunk_intra_working_dir(chunk_info)

  video_info = chunk_info.video_info
  H,W,C = video_info.resolution

  intra_fnames = get_chunk_intra_fnames(chunk_info)

  intra_info = []
  for intra_idx, intra_fname in enumerate(intra_fnames):
    intra_data = SimpleNamespace()
    intra_data.intra_fname = intra_fname
    intra_data.intra_decoding_param_fname = os.path.join(intra_working_dir, f'intra_dec_param_{intra_idx:06d}.txt')
    intra_data.out_bitstream_fname = os.path.join(intra_working_dir, f'intra_bs_{intra_idx:06d}.bin')
    intra_data.out_param_fname = os.path.join(intra_working_dir, f'intra_param_{intra_idx:06d}.txt')
    intra_data.out_recon_fname = get_chunk_intra_recon_fname(intra_working_dir, intra_idx)
    intra_data.cvc_intra_fname = os.path.join(intra_working_dir, f'intra_cvc_{intra_idx:06d}_{W}x{H}.yuv')
    intra_info.append(intra_data)
  return intra_info
 

def get_chunk_mix_intra_inter_fname(chunk_info):
  video_info = chunk_info.video_info
  H,W,C = video_info.resolution
  return os.path.join(chunk_info.working_dir, f'chunk_cvc_input_{chunk_info.chunk_idx}_{W}x{H}.yuv')

#if ZJU_VCM_SINGLE
def get_chunk_all_intra_fname(chunk_info):
  H,W,C = chunk_info.video_info.resolution
  return os.path.join(chunk_info.working_dir, f'chunk_cvc_input_{chunk_info.chunk_idx}_{W}x{H}_intra.yuv')
#endif

def get_chunk_cvc_output_recon_fname(chunk_info):
  H,W,C = chunk_info.video_info.resolution
  return os.path.join(chunk_info.working_dir, f'chunk_cvc_recon_{chunk_info.chunk_idx}_{W}x{H}.yuv')

def get_chunk_cvc_log_fname(chunk_info):
  return os.path.join(chunk_info.working_dir, f'chunk_cvc_log_{chunk_info.chunk_idx}.log')

def get_chunk_cvc_output_bs_fname(chunk_info):
  return os.path.join(chunk_info.working_dir, f'chunk_cvc_bs_{chunk_info.chunk_idx}.bin')

def get_video_cvc_output_bs_fname(video_working_dir):
  return os.path.join(video_working_dir, 'video_cvc_output_bs.bin')

def get_chunk_ima_output_dir(chunk_info):
  return os.path.join(chunk_info.working_dir, f'chunk_ima_recon_{chunk_info.chunk_idx}')
  
def get_chunk_ima_output_recon_fname(chunk_info, idx):
  chunk_ima_output_dir = get_chunk_ima_output_dir(chunk_info)
  return os.path.join(chunk_ima_output_dir, f'frame_{idx:06d}.png')

def get_chunk_ima_output_param_fname(chunk_info, idx):
  chunk_ima_output_dir = get_chunk_ima_output_dir(chunk_info)
  return os.path.join(chunk_ima_output_dir, f'adapter_param_{idx:06d}.txt')

def get_inner_output_bitstream_fname(working_dir):
  return os.path.join(working_dir, 'inner_codec_output.bin')

def get_inner_output_png_dir(working_dir):
  out_dir = os.path.join(working_dir, 'inner_codec_output_png')
  os.makedirs(out_dir, exist_ok = True)
  return out_dir

# determine if the encoder works in intra fallback mode
def isIntraFallbackMode(item):
  if item._is_video:
    return (item.args.quality + item.args.NNIntraQPOffset) > item.args.IntraFallbackQP
  else:
    # for image, NNIntraQPOffset does not apply
    return item.args.quality > item.args.IntraFallbackQP

# split a chunk intra frame file names
def get_chunk_intra_fnames(chunk_info):
  video_info = chunk_info.video_info
  if video_utils.is_yuv_video(chunk_info.video_info):
    # yuv video file
    intra_fnames = [video_info.frame_fnames + f':{idx}' for idx in \
      range(chunk_info.start_idx,chunk_info.end_idx+1,chunk_info.intra_period)]
  else:
    # directory
    intra_fnames = video_info.frame_fnames[ \
      chunk_info.start_idx:chunk_info.end_idx+1:chunk_info.intra_period]
  return intra_fnames

# interate frames
def get_chunk_frame_interator(chunk_info):
  video_info = chunk_info.video_info
  if video_utils.is_yuv_video(chunk_info.video_info):
    # yuv video file
    for idx in range(chunk_info.start_idx,chunk_info.end_idx+1):
      yield video_info.frame_fnames + f':{idx}' 
  else:
    # directory
    for fname in video_info.frame_fnames[chunk_info.start_idx:chunk_info.end_idx+1]:
      yield fname


########################################################
# other tools

# get frame information, e.g. QP values, from cvc bitstream
#if ZJU_VCM_SINGLE
def get_frame_info_from_cvc_bitstream(bs_fname, ctx, intra_yuv_fname):
#else
# def get_frame_info_from_cvc_bitstream(bs_fname, ctx):
#endif

  cvc_analyzer = os.path.join(ctx.input_args.cvc_dir, 'bin', 'DecoderAnalyserAppStatic')

#if ZJU_VCM_SINGLE
  if intra_yuv_fname == "":
    cmd = [
      cvc_analyzer,
      '-b', bs_fname,
    ]
  else:
    cmd = [
      cvc_analyzer,
      '-i', intra_yuv_fname,
      '-b', bs_fname,
    ]
#else
  # cmd = [
  #   cvc_analyzer, 
  #   '-b', bs_fname,
  # ]
#endif

  vcmrs.debug(cmd)
  err,outs = utils.run_process(cmd)
  assert err==0, f'CVC Encoding failed {outs}' 

  frame_info_dict = {}
  for line in outs.splitlines():
    m = re.search('^POC\s*(\d+).*QP\s*(-?\d+)', line)
    if m:
       poc = int(m.group(1))
       qp = int(m.group(2))
       frame_info = SimpleNamespace()
       frame_info.qp = qp
       frame_info_dict[poc] = frame_info
  return frame_info_dict


#if ZJU_VCM_SINGLE
def merge_intra_inter_frames(cvc_intra_fnames, chunk_info, output_seq_fname, nnic_rec_iha_fname, item):
#else
# def merge_intra_inter_frames(cvc_intra_fnames, chunk_info, output_seq_fname, item):
#endif
  '''
  Merge intra frame and inter frames to generate one 10-bit sequence in YUV color space
  '''
  # convert input to YUV420 10-bit
  input_yuv_fname = os.path.join(chunk_info.working_dir, f'chunk_input_{chunk_info.chunk_idx}.yuv')

  H,W,C = chunk_info.video_info.resolution
  if video_utils.is_yuv_video(chunk_info.video_info):
    cvt_option = f'select=between(n\,{chunk_info.start_idx}\,{chunk_info.end_idx})'

    cmd = [
       item.args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
       '-threads', '1', 
       '-f', 'rawvideo', 
       '-pix_fmt', video_utils.get_ffmpeg_pix_fmt(chunk_info.video_info),
       '-s', f'{W}x{H}',
       '-i', chunk_info.video_info.frame_fnames,
       '-f', 'rawvideo', 
       '-pix_fmt', 'yuv420p10le', 
       '-vf', cvt_option,
       input_yuv_fname]

    err = utils.start_process_expect_returncode_0(cmd, wait=True)
    assert err==0, 'Generating sequence in YUV format failed'
 
  else:
    # directory of png images
    # color format conversion
    input_frames = chunk_info.video_info.frame_fnames[chunk_info.start_idx:chunk_info.end_idx+1]
    data_utils.png_to_yuv420p10b(input_frames, input_yuv_fname)

  # merge cvc intra into input_yuv_fname
  frame_size = data_utils.get_frame_size_from_yuv(
    SourceWidth=W,
    SourceHeight=H,
    InputChromaFormat='420',
    InputBitDepth=10)

  intra_idx = 0
  
  #if ZJU_VCM_SINGLE
  with open(nnic_rec_iha_fname, 'wb') as of:
    for i_idx in range(len(cvc_intra_fnames)):
      # copy intra YUV to output
      with open(cvc_intra_fnames[i_idx], 'rb') as f:
        cvc_intra = f.read()
        of.write(cvc_intra)
  #endif

  with open(output_seq_fname, 'wb') as of:
    for frame_idx in range(chunk_info.start_idx, chunk_info.end_idx+1, chunk_info.intra_period):
      # copy intra YUV to output
      with open(cvc_intra_fnames[intra_idx], 'rb') as f:
        cvc_intra = f.read()
        of.write(cvc_intra)

      # copy inter yuv data to output
      inter_length = min(chunk_info.end_idx+1, frame_idx+chunk_info.intra_period) - frame_idx - 1
      with open(input_yuv_fname, 'rb') as f:
        f.seek(frame_size*(intra_idx*chunk_info.intra_period+1))
        inter_data = f.read(frame_size * inter_length)
        of.write(inter_data)
      
      intra_idx += 1

  return
 
# preprocessing video, for example downsampling if width or height exceed the threshold value
def preprocess_video(video_info, video_working_dir, item):
  video_info.scale_factor = 1

  H,W,C = video_info.resolution
  video_info.pad_h = H % 2 #0 if (H % 2)==0 else (2- (H % 2))
  video_info.pad_w = W % 2 #0 if (W % 2)==0 else (2- (W % 2))

  if (video_info.pad_h == 0) and (video_info.pad_w == 0): return

  # pad to even number of width and height if needed
  rescaled_input_dir = os.path.join(video_working_dir, 'rescaled_input')
  os.makedirs(rescaled_input_dir, exist_ok=True)
  rescaled_frame_fnames = []

  for fname in video_info.frame_fnames:
    bname = os.path.basename(fname)
    out_fname = os.path.join(rescaled_input_dir, bname)
    #new_resolution = data_utils.pad_image(fname, out_fname, video_info.pad_h, video_info.pad_w)
    data_utils.pad_image_even_ffmpeg(fname, out_fname, item)

    rescaled_frame_fnames.append(out_fname)

  video_info.resolution = H+video_info.pad_h, W+video_info.pad_w, C 
  video_info.frame_fnames = rescaled_frame_fnames

def init_enc_data(item, ctx):
  inner_wd = os.path.join(item.working_dir, 'inner')
  item.inner_working_dir = inner_wd
  os.makedirs(inner_wd, exist_ok=True)

def chunk_all_intra_frames_for_decoder(chunk_info_list, video_working_dir):
  output_intra_fname = os.path.join(video_working_dir, 'chunk_cvc_input_intra.yuv')
  with open(output_intra_fname, 'wb') as of:
    for chunk_info in chunk_info_list:
      # copy intra YUV to output
      chunk_yuv_fname = get_chunk_intra_recon_yuv_fname(get_chunk_intra_working_dir(chunk_info), 0, chunk_info.video_info)
      if os.path.exists(chunk_yuv_fname):
        with open(chunk_yuv_fname, 'rb') as f:
          cvc_intra = f.read()
          of.write(cvc_intra)
      else:
        output_intra_fname = ""
        break
  return output_intra_fname
# set extra information to item, these information may be used by the post processing components
#if ZJU_VCM_SINGLE
def set_extra_info_item(item, ctx, video_working_dir, chunk_info_list):
#else
# def set_extra_info_item(item, ctx, video_working_dir):
#endif
  # For IMA as post filter
  item.video_info.bit_depth = 10
  item.video_info.intra_indices = list(range(0, item.video_info.num_frames, item.IntraPeriod))
  bs_fname = get_video_cvc_output_bs_fname(video_working_dir)
  #if ZJU_VCM_SINGLE
  intra_yuv_fname = chunk_all_intra_frames_for_decoder(chunk_info_list, video_working_dir)
  #endif
  item.video_info.frame_qps = {k1:v1.qp for k1,v1 in get_frame_info_from_cvc_bitstream(bs_fname, ctx, intra_yuv_fname).items()}
  item.video_info.num_frames = len(item.video_info.frame_qps)
  item.intra_fallback = isIntraFallbackMode(item)

