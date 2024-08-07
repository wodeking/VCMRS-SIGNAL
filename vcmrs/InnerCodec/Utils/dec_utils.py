# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
# decoding utils
import os
import re
import glob
import asyncio
import traceback
import shutil
from types import SimpleNamespace

from . import bitstream_utils
from . import nnmanager_utils

import vcmrs
from vcmrs.Utils import data_utils
from vcmrs.Utils import utils

def process_file(item, ctx):
  # parse VCM bitstream
  if not item.args.VCMBitStructOn:
    inner_bitstream_fname = get_inner_bitstream_fname(item)
    bitstream_utils.parse_vcm_bitstream(item.fname, inner_bitstream_fname, item)
  else:
    inner_bitstream_fname = item.fname

  fname = inner_bitstream_fname 

  # check if the bitstream is for image compression
  if not item.args.VCMBitStructOn: # temporal solution. should change the header check for video.
    img_header = bitstream_utils.get_image_bitstream_header(fname)
    if img_header:
      # set inner out fname
      item.inner_out_fname = get_inner_output_fname(item, img=True)
      return process_image_bitstream(fname, item.inner_out_fname, img_header, item, ctx)

  # update properties in item
  item._is_video = True
  item._is_yuv_video = item.args.output_video_format=='YUV'
  item._is_dir_video = not item._is_yuv_video
  item.intra_fallback = True

  # initialization
  video_info = SimpleNamespace()
  video_info.fname = fname
  video_info.working_dir = item.working_dir #get_video_working_dir(fname)
  os.makedirs(video_info.working_dir, exist_ok=True)

  # parse bitstream, and set video_info
  # we need to trun this in intra fallback mode to parse video parameter set
  bitstream_demux(video_info, item, ctx)

  # this should be accessible in plugins 
  item.video_info = video_info

  # check if inner codec is VTM
  if  item.args.InnerCodec == 'VTM':
    process_VTM_encoded(video_info, item, ctx)
    return

  # check if the bitstream is in intra fallback mode, where the bitstream doesn't not contain NN IRAP NAL unit
  intra_fallback = bitstream_utils.is_intra_fallback_mode(fname)
  item.intra_fallback = intra_fallback

  if not intra_fallback:
    # process all intra frames
    asyncio.run(process_intra_frames(video_info, item, ctx))

    # mix intra bitstream and inter bitstreams
    mix_intra_inter_bitstreams(video_info, item)
    #if ZJU_VCM_SINGLE
    gen_cvc_input_video(item)
    #endif

  # decoding
  cvc_decode(video_info, item, ctx, intra_fallback=intra_fallback)

  # generate output
  gen_decoder_output(video_info, item, ctx, intra_fallback=intra_fallback)

  # run inter machine adapter
  #inter_machine_adapter(video_info, item, ctx, intra_fallback=intra_fallback)




def process_image_bitstream(fname, out_fname, header, item, ctx):
  # process image bitstream file
  intra_bs_fname = os.path.join(item.working_dir, 'intra_bitstream_fname.bin')
  with open(fname, 'rb') as f:
    data = f.read()
    with open(intra_bs_fname, 'wb') as of:
      of.write(data[4:]) # image header has fixed size 4 bytes

  # These parameters are required by NNManager, but not used for image decoding
  header.video_qp = 0 # Only needed when no model_id is provided in the bitstream
  header.IHA_flag = 0
  ret = nnmanager_utils.nnmanager_decode(
    bitstream_fname = intra_bs_fname, 
    recon_fname = out_fname, 
    intra_cfg = header,
    ctx = ctx)

# demux bitstream, get video params, split intra
def bitstream_demux(video_info, item, ctx):
  # call bistream demux
  bs_demux_app = os.path.join(ctx.input_args.cvc_dir, 'bin', 'FrameSplitterNNAppStatic')
  intra_bs_prefix = get_intra_bitstream_prefix(item)
  intra_cfg_prefix = get_intra_config_prefix(item)
  ima_param_prefix = get_ima_param_prefix(item)

  if not item.args.VCMBitStructOn:
    cmd = [ 
      bs_demux_app, 
      '-b', video_info.fname,
      '-o', intra_bs_prefix,
      '-c', intra_cfg_prefix,
      '-s', ima_param_prefix
    ]
  else:
    vcmrs.log('VCM bistream strucutre is on at decode stage 1.\n')
    rsd_fname = get_rsd_fname(item)
    cvd_fname = get_cvd_fname(item)
    cmd = [ 
      bs_demux_app, 
      '-b', video_info.fname,
      '-r', rsd_fname, # restoration data file name
      '-v', cvd_fname # coded video data file name
    ]
    video_info.fname = cvd_fname

  vcmrs.debug(cmd)
  err, outs = utils.start_process(cmd, wait=True, return_out=True)
  assert err==0, f"The format of bitstream file {video_info.fname} is incorrect!"

  # collect the rsd data.
  if item.args.VCMBitStructOn:
    with open(rsd_fname, 'rb') as f:
      item.parameters = bitstream_utils.parse_parameters_bitstream(f)

  # parse output to get width and height of the intra pictures
  width,height = parse_demux_output(outs)
  video_params = SimpleNamespace()
  video_params.width = width
  video_params.height = height 
  video_info.params = video_params
  
  # get intra bitstreams and intra params
  video_info.intra_bitstream_fnames = sorted(glob.glob(intra_bs_prefix+'*.bin'))
  video_info.intra_param_fnames = sorted(glob.glob(intra_cfg_prefix+'*.txt'))

  # get output rescaling factor
  video_info.scale_factor = 1
  #video_info.scale_factor = get_rescale_factor(cfg, video_info)
  pad_h, pad_w, ima_off = get_ima_pad_information(item, video_info)
  video_info.pad_h = pad_h
  video_info.pad_w = pad_w
  video_info.ima_off = ima_off

  # update resolution for the output of inner codec
  video_info.resolution = (video_info.params.height+video_info.pad_h,
      video_info.params.width + video_info.pad_w, 3)
  item.video_info = video_info

#if ZJU_VCM_SINGLE
async def lic_dec_worker(queue):
#else
# async def lic_dec_worker(queue, cvc_queue):
#endif
  try: 
    while True:
      intra_data, item, ctx = await queue.get()
      await lic_decode_async(intra_data, item, ctx)
     
      #await cvc_queue.put((intra_cvc_fname, intra_bitstream_fname))
      #if ZJU_VCM_SINGLE
      # await cvc_queue.put((intra_data, ctx))
      #endif
      queue.task_done()
  except Exception as e: 
      callstack = f"\n-------Formatted stack-----------\n{traceback.format_exc()}"
      callstack += "\n-------Call queue (Most recent call last)-----------\n'" + '\n'.join([line.strip() for line in traceback.format_stack()])
      vcmrs.error(callstack)
      #raise e
  finally:
    queue.task_done()
    return  -1
 
async def cvc_enc_worker(queue):
  while True:
    intra_data, ctx = await queue.get()

    # intra cvc input
    intra_cvc_fname = get_intra_cvc_input_fname(intra_data)
    # intra cvc output
    intra_bitstream_fname = get_intra_cvc_bitstream_fname(intra_data)
    intra_recon_fname = get_intra_cvc_recon_fname(intra_data)
 
    cvc_dir = ctx.input_args.cvc_dir
    cvc_encoder=os.path.join(cvc_dir, 'bin', 'EncoderAppStatic')

    if intra_data.cvc_config == "LowDelay":
      cvc_encoder_cfg1=os.path.join(cvc_dir, 'cfg', 'encoder_lowdelay_vtm.cfg')
    else:
      cvc_encoder_cfg1=os.path.join(cvc_dir, 'cfg', 'encoder_randomaccess_vtm.cfg')
      
    cvc_encoder_cfg2=os.path.join(cvc_dir, 'cfg', 'lossless', 'lossless_intra.cfg')
    W = intra_data.video_params.width
    H = intra_data.video_params.height

    cmd = [ 
      cvc_encoder, 
      '-c', cvc_encoder_cfg1,
      '-c', cvc_encoder_cfg2,
      f'--InputFile={intra_cvc_fname}', 
      f'--SourceWidth={W}',
      f'--SourceHeight={H}',
      f'--InputBitDepth=10',
      f'--BitstreamFile={intra_bitstream_fname}',
      f'--FramesToBeEncoded=1',
      f'--FrameRate=1',
      '-o', intra_recon_fname,
    ]

    vcmrs.debug('##########################')
    # vcmrs.debug(cmd)
    await utils.start_process_async(cmd)
    queue.task_done()
 

async def process_intra_frames(video_info, item, ctx):

  coding_idx = 0

  intra_dec_queue = asyncio.Queue()
  #if ZJU_VCM_SINGLE
  #cvc_enc_queue = asyncio.Queue()
  #endif

  # prepare LIC worker
  #if ZJU_VCM_SINGLE
  lic_task = asyncio.create_task(lic_dec_worker(intra_dec_queue))
  #else
  # lic_task = asyncio.create_task(lic_dec_worker(intra_dec_queue, cvc_enc_queue))

  # prepare cvc encoder
  # cvc_enc_task = asyncio.create_task(cvc_enc_worker(cvc_enc_queue))

  # Only param of InnerCodec for now
  # cvc_config = item.get_parameter("InnerCodec").decode()
  #endif
  # collect intra frames first 
  for intra_idx, (intra_bs_fname, intra_param_fname) in enumerate(
      zip(video_info.intra_bitstream_fnames, video_info.intra_param_fnames)):
    intra_data = SimpleNamespace()
    intra_data.idx = intra_idx
    intra_data.video_params = video_info.params
    intra_data.bs_fname = intra_bs_fname
    intra_data.param_fname = intra_param_fname
    #if ZJU_VCM_SINGLE
    # intra_data.cvc_config = cvc_config
    #endif
    intra_dec_queue.put_nowait((intra_data, item, ctx))

  # process intra frames
  await intra_dec_queue.join()
  #if ZJU_VCM_SINGLE
  # await cvc_enc_queue.join()
  #endif

  lic_task.cancel()
  #if ZJU_VCM_SINGLE
  # cvc_enc_task.cancel()
  #endif

  # wait until all workers are cancelled
  #if ZJU_VCM_SINGLE
  await asyncio.gather(lic_task, return_exceptions=True)
  #else
  # await asyncio.gather(lic_task, cvc_enc_task, return_exceptions=True)
  #endif

# process a intra frame
async def lic_decode_async(intra_data, item, ctx):
  intra_recon_fname = get_intra_recon_fname(intra_data.idx, item)
  #if ZJU_VCM_SINGLE
  intra_cvc_fname = get_chunk_all_intra_fname(intra_data)
  #else
  # intra_cvc_fname = get_intra_cvc_input_fname(intra_data)
  #endif
  intra_params = bitstream_utils.get_params(intra_data.param_fname)
  # this shall not be used to intra codec
  intra_params.video_qp = -1
  intra_params.picture_width = intra_data.video_params.width
  intra_params.picture_height = intra_data.video_params.height

  ret = await nnmanager_utils.nnmanager_decode_async(
    intra_data.bs_fname, 
    recon_fname = intra_recon_fname, 
    cvc_intra_fname = intra_cvc_fname,
    intra_cfg = intra_params, 
    ctx = ctx)

  vcmrs.debug(f'=================== ret={ret}')
  return ret


# mix intra bitstreams with inter bitstreams
def mix_intra_inter_bitstreams(video_info, item):
  mix_output_fname = get_mix_bitstream_fname(item)
  intra_cvc_bs_prefix = get_intra_cvc_bs_prefix(item)
  frame_mixer = os.path.join(item.args.cvc_dir, 'bin', 'FrameMixerDecAppStatic')

  cmd = [ 
    frame_mixer, 
    '-b', video_info.fname, 
    '-i', intra_cvc_bs_prefix, 
    '-o', mix_output_fname,
  ]

  print(cmd)

  err = utils.start_process(cmd, wait=True)
  assert err==0, 'Intra inter bitstream mixer failed!'

 

# CVC decode
#   intra_fallback: in intra fallback mode, the input bitstream is a normal CVC bitstream
def cvc_decode(video_info, item, ctx, intra_fallback=False):
  if intra_fallback:
    mix_output_fname = video_info.fname
  else:
    mix_output_fname = get_mix_bitstream_fname(item)
  cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)
  #if ZJU_VCM_SINGLE
  cvc_input_intra_fname = get_cvc_input_intra_fname(item)
  #endif
  cvc_decoder=os.path.join(ctx.input_args.cvc_dir, 'bin', 'DecoderAppStatic')

#if ZJU_VCM_SINGLE
  if intra_fallback:
    cmd = [
      cvc_decoder,
      f'--BitstreamFile={mix_output_fname}',
      f'--ReconFile={cvc_dec_recon_fname}',
    ]
  else:
    cmd = [
      cvc_decoder,
      f'--IntraFile={cvc_input_intra_fname}',
      f'--BitstreamFile={mix_output_fname}',
      f'--ReconFile={cvc_dec_recon_fname}',
    ]
#else
  # cmd = [ 
  #   cvc_decoder, 
  #   f'--BitstreamFile={mix_output_fname}', 
  #   f'--ReconFile={cvc_dec_recon_fname}',
  # ]
#endif

  err, outs = utils.start_process(cmd, wait=True, return_out=True)
  parse_cvc_output(video_info, outs)

  assert err==0, 'CVC decoder failed!'
 
# VTM as inner codec
def process_VTM_encoded(video_info, item, ctx):
  # decode
  bs_fname = video_info.fname
  cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)
  cvc_decoder=os.path.join(ctx.input_args.cvc_dir, 'bin', 'DecoderAppStatic')
  vcm_based_upscaling_flag = get_parameter(item)
  
  cmd = [ 
    cvc_decoder, 
    f'--BitstreamFile={bs_fname}', 
    f'--ReconFile={cvc_dec_recon_fname}',
    
  ]
  
  if vcm_based_upscaling_flag == 0:
     # Perform upscaling at VTM level
    cmd.extend(['--UpscaledOutput=2'])
    

  err, outs = utils.start_process(cmd, wait=True, return_out=True)
  parse_cvc_output(video_info, outs)

  assert err==0, 'CVC decoder failed!'

  resolution = (video_info.params.height, video_info.params.width, 3) #HWC
  video_info.resolution = resolution # this should be accessible in plugins 

  # image
  if video_info.num_frames==1 and item.args.single_frame_image:
    cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item)
    item.inner_out_fname = get_inner_output_fname(item, img=True)
    output_recon_fname = item.inner_out_fname

    data_utils.yuv420p10b_to_png_ffmpeg(
       cvc_dec_recon_fname+f':0', 
       resolution,
       cvc_dec_recon_png_fname,
       item=item)

    # unpad if necessary
    unpad_frame(
      cvc_dec_recon_png_fname, 
      output_recon_fname, 
      resolution,
      pad_h = video_info.pad_h,
      pad_w = video_info.pad_w,
      ffmpeg=True,
      item = item
      ) 
    return
  # video 
  # check output format
  if item.args.output_video_format=='YUV':
    item.inner_out_fname = get_inner_output_fname(item, format='YUV')
    # directory copy YUV output
    shutil.copy(cvc_dec_recon_fname, item.inner_out_fname)

  else: 
    item.inner_out_fname = get_inner_output_fname(item, format='PNG')
    # convert YUV to png
    for idx in range(video_info.num_frames):
      output_recon_fname = get_output_fname(idx, item)
      cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item, idx=idx)
      # perform color conversion using ffmpeg when VTM is used as inner codec
      data_utils.yuv420p10b_to_png_ffmpeg(
          cvc_dec_recon_fname+f':{idx}', 
          resolution, 
          #(video_info.params.height, video_info.params.width, 3), #HWC, resolution
          cvc_dec_recon_png_fname, 
          item = item)

      # unpad if needed
      unpad_frame(
        cvc_dec_recon_png_fname,
        output_recon_fname,
        resolution = (video_info.params.height, video_info.params.width, 3), #HWC
        pad_h = video_info.pad_h,
        pad_w = video_info.pad_w,
        ffmpeg=True, 
        item = item)
 
def gen_decoder_output(video_info, item, ctx, intra_fallback):
  cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)

  # image
  if video_info.num_frames==1 and item.args.single_frame_image:
    gen_decoer_output_img_from_video_bitstream(video_info, item, ctx, intra_fallback)
    return

  #video
  if item.args.output_video_format=='YUV':
    item.inner_out_fname = get_inner_output_fname(item, format='YUV')
    # directory copy YUV output
    if video_info.ima_off: 
      if intra_fallback:
        shutil.copy(cvc_dec_recon_fname, item.inner_out_fname)
      else:
        mix_intra_inter_yuv(cvc_dec_recon_fname, video_info, item)
    else: 
      inner_out_fname = item.inner_out_fname
      # save output png to temporary directory 
      item.inner_out_fname = os.path.join(item.working_dir, "inner_out_png")
      os.makedirs(item.inner_out_fname, exist_ok=True)
      apply_inter_machine_adapter(video_info, item, ctx, intra_fallback)
      # convert png to YUV
      png_fnames = sorted(glob.glob(os.path.join(item.inner_out_fname, '*.png')))
      data_utils.png_to_yuv420p_ffmpeg(png_fnames, inner_out_fname, bitdepth=10, item=item)
      item.inner_out_fname = inner_out_fname
  else: 
    item.inner_out_fname = get_inner_output_fname(item, format='PNG')
    apply_inter_machine_adapter(video_info, item, ctx, intra_fallback)

# mix LIC intra output
def mix_intra_inter_yuv(cvc_dec_recon_fname, video_info, item):
  out_fname = item.inner_out_fname

  # generate output directory
  os.makedirs(os.path.dirname(item.inner_out_fname), exist_ok=True)

  yuv_fnames = []
  intra_idx = 0 #intra frame idx

  for idx in range(video_info.num_frames):
    output_recon_fname = get_output_fname(idx, item)

    if idx in video_info.intra_indices:
      # LIC encoded intra frame
      intra_recon_fname = get_intra_recon_fname(intra_idx, item)
      intra_recon_fname_pad = intra_recon_fname+'_pad.png'
      intra_recon_fname_yuv = intra_recon_fname+'.yuv'

      # unpad if needed
      unpad_frame(
        intra_recon_fname,
        intra_recon_fname_pad,
        resolution = (video_info.params.height, video_info.params.width, 3), #HWC
        pad_h = video_info.pad_h,
        pad_w = video_info.pad_w,
        ffmpeg=False, 
        item = item)

      # convert to YUV
      data_utils.png_to_yuv420p_ffmpeg(
        [intra_recon_fname_pad], 
        intra_recon_fname_yuv, 
        bitdepth=10, 
        item=item)
      yuv_fnames.append(intra_recon_fname_yuv)

      intra_idx += 1
    else:
      # get cvc YUV output
      out_yuv_frame_fname= cvc_dec_recon_fname + f':{idx}'

      yuv_fnames.append(out_yuv_frame_fname)

  # concat all yuv files
  W = video_info.params.width
  H = video_info.params.height
  data_utils.concat_yuvs(yuv_fnames, width=W, height=H, bitdepth=10, num_overlap_frames=0, out_fname=item.inner_out_fname)  

def apply_inter_machine_adapter(video_info, item, ctx, intra_fallback):
  cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)

  # generate output directory
  os.makedirs(item.inner_out_fname, exist_ok=True)

  #intra frame idx
  intra_idx = 0
  for idx in range(video_info.num_frames):
    output_recon_fname = get_output_fname(idx, item)

    if idx in video_info.intra_indices and not intra_fallback:
      # LIC encoded intra frame
      #frame_idx = video_info.frame_indices[coding_idx]
      intra_recon_fname = get_intra_recon_fname(intra_idx, item)
      shutil.copy(intra_recon_fname, output_recon_fname)
      intra_idx += 1
    else:
      # VTM encoded frame
      cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item, idx=idx)
      # prepare parameters
      param = SimpleNamespace()
      param.qp = video_info.frame_qps[idx]
      param.intra_fallback = intra_fallback
      # prepare video info
      ima_video_info = SimpleNamespace()
      ima_video_info.resolution = (video_info.params.height, video_info.params.width, 3) #HWC
      ima_video_info.bit_depth = 10
      ima_video_info.chroma_format='420'
      ima_video_info.color_space='yuv'

      # hardcoded inter machine adapter switch
      # todo: IMA flag should be received from the bitstream
      if video_info.ima_off:
        # perform color conversion using opencv
        data_utils.yuv420p10b_to_png(
          cvc_dec_recon_fname+f':{idx}', 
          ima_video_info.resolution,
          cvc_dec_recon_png_fname)

      else:
        # perform ima
        nnmanager_utils.nnmanager_inter_machine_adapter(
          input_fname = cvc_dec_recon_fname+f':{idx}', 
          output_fname = cvc_dec_recon_png_fname,
          gt_fname = None,
          param = param, 
          video_info = ima_video_info, 
          ctx = ctx)

      # unpad if needed
      unpad_frame(
        cvc_dec_recon_png_fname,
        output_recon_fname,
        resolution = (video_info.params.height, video_info.params.width, 3), #HWC
        pad_h = video_info.pad_h,
        pad_w = video_info.pad_w,
        ffmpeg=False, 
        item = item)

 
def gen_decoer_output_img_from_video_bitstream(video_info, item, ctx, intra_fallback):
    # image decoding
    cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item)
    item.inner_out_fname = item.get_stage_output_fname_decoding(stage='inner', img=True)
    output_recon_fname = item.inner_out_fname

    H = video_info.params.height
    W = video_info.params.width
    if intra_fallback:
      # fallback mode
      # perform color conversion
      cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)
      data_utils.yuv420p10b_to_png(
          cvc_dec_recon_fname+f':0', 
          (H, W, 3), #HWC
          cvc_dec_recon_png_fname)
    else:
      # intra frame
      intra_recon_fname = get_intra_recon_fname(0, item)
      shutil.copy(intra_recon_fname, cvc_dec_recon_png_fname)

    # unpad if necessary
    unpad_frame(
      cvc_dec_recon_png_fname, 
      output_recon_fname, 
      (H,W,3),
      pad_h = video_info.pad_h,
      pad_w = video_info.pad_w,
      ffmpeg=False,
      item = item
      ) 



# apply inter machine adapter
#   intra_fallback: in intra fallback mode, inter machine adapter is also applied to the intra frames
def inter_machine_adapter_osbsolete(video_info, item, ctx, intra_fallback=False):
  cvc_dec_recon_fname = get_cvc_dec_recon_fname(item)

  if video_info.num_frames==1 and item.args.single_frame_image:
    # image decoding
    cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item)
    item.inner_out_fname = item.get_stage_output_fname_decoding(stage='inner', img=True)
    output_recon_fname = item.inner_out_fname

    H = video_info.params.height
    W = video_info.params.width
    if intra_fallback:
      # fallback mode
      # perform color conversion
      if item.args.InnerCodec == 'VTM':
        data_utils.yuv420p10b_to_png_ffmpeg(
            cvc_dec_recon_fname+f':0', 
            (H, W, 3), #HWC
            cvc_dec_recon_png_fname,
            item)
      else: 
        data_utils.yuv420p10b_to_png(
          cvc_dec_recon_fname+f':0', 
          (H, W, 3), #HWC
          cvc_dec_recon_png_fname)
    else:
      # intra frame
      intra_recon_fname = get_intra_recon_fname(0, item)
      shutil.copy(intra_recon_fname, cvc_dec_recon_png_fname)

    # unpad if necessary
    unpad_frame(
      cvc_dec_recon_png_fname, 
      output_recon_fname, 
      (H,W,3),
      pad_h = video_info.pad_h,
      pad_w = video_info.pad_w,
      ffmpeg = item.args.InnerCodec == 'VTM',
      item = item)
    return

  # video decoding

  # generate output directory
  os.makedirs(item.inner_out_fname, exist_ok=True)

  #intra frame idx
  intra_idx = 0
  for idx in range(video_info.num_frames):
    output_recon_fname = get_output_fname(idx, item)

    if idx in video_info.intra_indices and not intra_fallback:
      # LIC encoded intra frame
      #frame_idx = video_info.frame_indices[coding_idx]
      intra_recon_fname = get_intra_recon_fname(intra_idx, item)
      shutil.copy(intra_recon_fname, output_recon_fname)
      intra_idx += 1
    else:
      # VTM encoded frame
      cvc_dec_recon_png_fname = get_cvc_dec_recon_png_fname(item, idx=idx)
      # prepare parameters
      param = SimpleNamespace()
      param.qp = video_info.frame_qps[idx]
      param.intra_fallback = intra_fallback
      # prepare video info
      ima_video_info = SimpleNamespace()
      ima_video_info.resolution = (video_info.params.height, video_info.params.width, 3) #HWC
      ima_video_info.bit_depth = 10
      ima_video_info.chroma_format='420'
      ima_video_info.color_space='yuv'

      # hardcoded inter machine adapter switch
      # todo: IMA flag should be received from the bitstream
      if item.args.InnerCodec != 'VTM':
        nnmanager_utils.nnmanager_inter_machine_adapter(
          input_fname = cvc_dec_recon_fname+f':{idx}', 
          output_fname = cvc_dec_recon_png_fname,
          gt_fname = None,
          param = param, 
          video_info = ima_video_info, 
          ctx = ctx)
      else:
        # perform color conversion using ffmpeg when VTM is used as inner codec
        data_utils.yuv420p10b_to_png_ffmpeg(
          cvc_dec_recon_fname+f':{idx}', 
          ima_video_info.resolution,
          cvc_dec_recon_png_fname,
          item)

      # unpad if needed
      unpad_frame(
        cvc_dec_recon_png_fname,
        output_recon_fname,
        pad_h = video_info.pad_h,
        pad_w = video_info.pad_w,
        ffmpeg = item.args.InnerCodec == 'VTM',
        item = item)

    # rescale if needed
    #if video_info.scale_factor != 1:
    #  # rescale image
    #  data_utils.resize_image(output_recon_fname, output_recon_fname, 1/video_info.scale_factor)

def unpad_frame(in_fname, out_fname, resolution, pad_h, pad_w, ffmpeg, item):
    # unpad if necessary
    if (pad_h==0) and (pad_w==0):
      shutil.copy(in_fname, out_fname)
    else:
      H,W,C = resolution
      if ffmpeg:
        data_utils.unpad_image_ffmpeg(
          in_fname,
          out_fname, 
          width = W - pad_w, 
          height = H - pad_h,
          item = item)
      else:
        data_utils.unpad_image(in_fname, 
          out_fname, 
          pad_h, 
          pad_w) 
 
###################################################
# intermediate file names


def get_inner_bitstream_fname(item):
  fname = os.path.join(item.working_dir, 'inner', f'{item._bname}.bin')
  os.makedirs(os.path.dirname(fname), exist_ok=True)
  return fname
  #return item.inner_in_fname

def get_video_param_fname(video_info):
  return os.path.join(video_info.working_dir, f'video_params.txt')

def get_intra_cvc_bitstream_fname(intra_data):
  working_dir = os.path.dirname(intra_data.bs_fname)
  return os.path.join(working_dir, f'intra_cvc_bitstream_{intra_data.idx:06d}.bin')

def get_intra_cvc_recon_fname(intra_data):
  working_dir = os.path.dirname(intra_data.bs_fname)
  width = intra_data.video_params.width
  height = intra_data.video_params.height
  return os.path.join(working_dir, f'intra_cvc_recon_{width}x{height}_{intra_data.idx:06d}.yuv')

def get_intra_cvc_bs_prefix(item):
  return os.path.join(item.working_dir, 'intra_cvc_bitstream_')

def get_intra_recon_fname(idx, item):
  return os.path.join(item.working_dir, f'intra_recon_{idx:06d}.png')

def get_intra_cvc_input_fname(intra_data):
  working_dir = os.path.dirname(intra_data.bs_fname)
  width = intra_data.video_params.width
  height = intra_data.video_params.height

  return os.path.join(working_dir, f'intra_cvc_input_{width}x{height}_{intra_data.idx}.yuv')

#if ZJU_VCM_SINGLE
def get_chunk_all_intra_fname(intra_data):
  working_dir = os.path.dirname(intra_data.bs_fname)
  return os.path.join(working_dir, f'chunk_cvc_input_{intra_data.idx}_intra.yuv')
#endif

def get_mix_bitstream_fname(item):
  return os.path.join(item.working_dir, 'mix_bitstream.bin')

def get_cvc_dec_recon_fname(item):
  return os.path.join(item.working_dir, 'cvc_dec_recon.yuv')

def get_cvc_dec_recon_png_fname(item, idx=0):
  return os.path.join(item.working_dir, f'cvc_dec_recon_{idx}.png')

def get_intra_bitstream_prefix(item):
  return os.path.join(item.working_dir, 'intra_bs_')

def get_intra_config_prefix(item):
  return os.path.join(item.working_dir, 'intra_config_')

def get_ima_param_prefix(item):
  return os.path.join(item.working_dir, 'adapter_param_')

def get_ima_param_fname(item, frame_idx):
  ima_param_prefix = get_ima_param_prefix(item)
  return ima_param_prefix + f"{frame_idx:06d}.txt"

def get_rsd_fname(item):
  os.makedirs(item.working_dir, exist_ok=True)
  return os.path.join(item.working_dir, f'tmp_rsd.bin')

def get_cvd_fname(item):
  os.makedirs(item.working_dir, exist_ok=True)
  return os.path.join(item.working_dir, f'tmp_cvd.bin')

def get_output_fname(frame_idx, item):
  return os.path.join(item.inner_out_fname, 
     f"frame_{frame_idx:06d}.png")

def get_inner_output_fname(item, img=False, format='YUV'):
    sd = os.path.join(item.working_dir, 'inner')
    os.makedirs(sd, exist_ok=True)
    if img: 
      return os.path.join(sd, os.path.splitext(item._bname)[0]+'.png')

    if format=='YUV':
      out_fname = os.path.join(sd, os.path.splitext(item._bname)[0]+'.yuv')
    else:
      out_fname = os.path.join(sd, item._bname)
      os.makedirs(out_fname, exist_ok=True)
    return out_fname

#if ZJU_VCM_SINGLE
def get_cvc_input_intra_fname(item):
  return os.path.join(item.working_dir, f'chunk_cvc_input_intra.yuv')

def gen_cvc_input_video(item):
  cvc_input_intra_fname = get_cvc_input_intra_fname(item)
  intra_nums = 0
  while(True):
    temp_path = os.path.join(item.working_dir, 'chunk_cvc_input_{}_intra.yuv'.format(intra_nums))
    if os.path.exists(temp_path): intra_nums += 1
    else: break
  # lic_output_fnames = glob.glob(os.path.join(item.working_dir, r'chunk_cvc_input_**_layer0.yuv'), recursive=True)

  with open(cvc_input_intra_fname, 'wb') as of:
    for intra_idx in range(intra_nums):
      # copy intra YUV to output
      lic_output_fname = os.path.join(item.working_dir, 'chunk_cvc_input_{}_intra.yuv'.format(intra_idx))
      with open(lic_output_fname, 'rb') as f:
        cvc_input_intra_fname = f.read()
        of.write(cvc_input_intra_fname)
#endif

# input file handling
def get_input_files(args):
  input_files = []
  for fname in args.input_files:
    if  os.path.isdir(fname):
      input_files += sorted(glob.glob(os.path.join(fname, '*')))
    else:
      input_files.append(fname)
  return input_files

def parse_demux_output(outs):
  for line in outs.splitlines():
    if line.startswith('width'):
       width = int(line.split()[1])
    if line.startswith('height'):
       height = int(line.split()[1])
       return width, height
  assert False, 'width and height are not found in the output of demux'
  return None

def parse_cvc_output(video_info, outs):
  intra_indices = []
  qps = []
  for line in outs.splitlines():
    if line.startswith('POC '):
      m = re.search('^POC\s+(\d+).*?\(\s+([A-Z_]+).*?QP(.*?)\)', line)
      idx = int(m.group(1))
      pic_type = m.group(2).strip()
      qp = int(m.group(3))
      if pic_type=='CRA' or pic_type.startswith('IDR'):
        intra_indices.append(idx)
      if idx > len(qps)-1:
        qps += [None] * (idx +1 - len(qps))
      qps[idx] = qp

  video_info.num_frames = len(qps)
  video_info.frame_qps = qps
  video_info.intra_indices = intra_indices

def get_ima_pad_information(item, video_info):
  ima_param_fname = get_ima_param_fname(item, 0)
  pad_h = 0
  pad_w = 0
  ima_off = 1
  if os.path.exists(ima_param_fname):
    ima_param = bitstream_utils.get_params(ima_param_fname)
    pad_h = (ima_param.IMA_model_id % 64)// 8
    pad_w = ima_param.IMA_model_id % 8
    ima_off = ima_param.IMA_model_id // 64
  return pad_h, pad_w, ima_off

# ==================================================================
def get_parameter(item):
  # sequence level parameter
  vcm_based_upscaling_flag = -1
  param_data = item.get_parameter('SpatialResample')
  if param_data is not None:
    assert len(param_data) == 1, f'received parameter data is not correct: {param_data}'
    vcm_based_upscaling_flag = param_data[0]
     
  return vcm_based_upscaling_flag