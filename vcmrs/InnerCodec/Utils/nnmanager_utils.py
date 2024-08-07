# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
import os
from types import SimpleNamespace
import json
import zmq
import zmq.asyncio
import tempfile
import time

import vcmrs
from vcmrs.Utils import utils

###############################################
# NNManager utils
def nnmanager_start_(ctx): # internal implementation
  # Singleton NNManager to avoid bugs
  config_path="run_config.yaml"
  if getattr(ctx.inner_ctx, "nnmanager_proc", -1 ) == -1: 
    mng_dir = os.path.join(os.path.dirname(__file__), '..', 'NNManager', 'Main')
    nnmanager_path = os.path.join(mng_dir, 'nnmanager.py')
    info_fp, info_file_fname = tempfile.mkstemp(text=True)
    cmd = [
      'python', 
      nnmanager_path,
      '--port', str(ctx.input_args.port),
      '--info_file', info_file_fname,
      '--IntraCodec', ctx.input_args.IntraCodec,
      '--NNConfigs', config_path,
      '--logfile', ctx.input_args.logfile
    ]
    if ctx.input_args.debug:
      cmd.append('--debug')
    try:
      ctx.inner_ctx.nnmanager_proc = utils.start_process(cmd)
      if ctx.inner_ctx.nnmanager_proc is None:
        os.remove(info_file_fname)
        ctx.inner_ctx.nnmanager_proc = -1
        return False
    except:
      os.remove(info_file_fname)
      return False

    for idx in range(120):
      with open(info_file_fname, 'r') as f:
        try:
          info = json.load(f)
          if 'port' in info.keys():
            vcmrs.log(f'Connect to NNManager on port {info["port"]}')
            ctx.input_args.port = info['port']
            os.remove(info_file_fname)
            return True
        except json.decoder.JSONDecodeError:
          pass
      time.sleep(5)    
    # failed: terminate zombie-process
    try:
      ctx.inner_ctx.nnmanager_proc.terminate() 
      time.sleep(1)  
      ctx.inner_ctx.nnmanager_proc.kill() 
      os.remove(info_file_fname)
    finally:
      ctx.inner_ctx.nnmanager_proc = -1
    return False

def nnmanager_start(ctx):
  for i in range(10):
    nnmanager_ready = nnmanager_start_(ctx)
    if nnmanager_ready: return
    vcmrs.log(f'NNManager restart: {i}') # PUT
    time.sleep(2)

  assert nnmanager_ready, "NNManager cannot be started"

def nnmanager_stop(ctx):
  if getattr(ctx.inner_ctx, "nnmanager_proc", -1) != -1:
    time.sleep(1)  
    ctx.inner_ctx.nnmanager_proc.kill()
    ctx.inner_ctx.nnmanager_proc = -1

def nnmanager_request(cmd, ctx):
  '''
  Send a request to NNManager
  
  Args:
    cmd: Command, a object of SimpleNamesapce
  '''
  msg = json.dumps(cmd.__dict__).encode()

  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://localhost:%s" % ctx.input_args.port)

  # send message and wait for response
  socket.send(msg)
  vcmrs.debug(f'Message to NNManager: {msg}')

  # wiat for reply
  msg = socket.recv().decode('utf-8')
  vcmrs.log(f'Message received from NNManager: {msg}')
  reply = json.loads(msg, object_hook=lambda d: SimpleNamespace(**d))
  return reply

async def nnmanager_encode_async(input_fname, 
    bitstream_fname, 
    ctx,
    recon_fname=None,
    cvc_intra_fname=None,
    param_fname=None,
    intra_cfg=None,
    video_info=None):
  '''
  Intra frame coding
  
  Args:
    bitstream_fname: bitstream file name
    recon_fname: reconstruction file name
    intra_cfg: intra codec and intra human adapter configuration. 
      intra_human_adapter_settings: tbd
    video_info: video information for YUV video file

  Command sent to the NNManager is in SimpleNamespace data type with the following fields: 

  Command:
  --------
    action: "encode"
    input_fname: input file. If the file is a png image, input_fname is the file name. If the input is a frame from a YUV sequence, input_fname has format of <yuv_sequnce_file_name:frame_index. For example, "video1_640x320.yuv:20" is the 21th frame of the yuv file. The pixel format, e.g., 420 or 444, 10-bit or bit, is specified in field video_info.
    bitstream_fname: output bitstream file that contains the slice data in the intra NALU. 
    param_fname: the file that contains the paramters to write to the slice header in the intra NALU 
    cvc_intra_fname: the file name for intra human adapter (CVC pre-filter) processed frame. The format shall be in yuv420 10-bit. 
    recon_fname: the reconstructed frame in png format from the LIC codec.
    video_info: video information. This information is given when the input file is a yuv sequence. Otherwise, it is null. 
      frame_rate: frame rate
      color_space: 'rgb' or 'yuv'
      num_frames: number of frames
      bit_depth: 8 or 10
      chroma_format: only valide for yuv color space. 420 or 444
      resolution: frame resolution for yuv sequnce input. In C H W format.
      frame_fnames: for yuv sequence, it is the input yuv file name
       
    intra_cfg: configurations to the LIC codec. 
      video_qp: video quality, for example, QP value

  Reply:
  ------
    error: error code. 0 indicates success. 
    When the error code is 0, the following fields are returned to be signaled to decoder
      model_id: 0-7, indicate teh model_id

  '''
  
  # prepare message
  cmd = SimpleNamespace()
  cmd.action = 'encode'
  cmd.input_fname = input_fname 
  cmd.bitstream_fname = bitstream_fname
  cmd.recon_fname = recon_fname
  cmd.cvc_intra_fname = cvc_intra_fname
  cmd.param_fname = param_fname

  # flatten the cmd object, since only flat object can be jsonfied. 
  if intra_cfg:
    cmd.intra_cfg = json.dumps(intra_cfg.__dict__)
  else:
    cmd.intra_cfg = None

  if video_info:
    cmd.video_info = json.dumps(video_info.__dict__)
  else:
    cmd.video_info = None
  
  # set request 
  reply = await nnmanager_request_async(cmd, ctx)

  # check error code
  return reply

async def nnmanager_request_async(cmd, ctx):
  '''
  Send a request to NNManager
  
  Args:
    cmd: Command, a object of SimpleNamesapce
  '''
  async with ctx.inner_ctx.nnmanager_lock:
    msg = json.dumps(cmd.__dict__).encode()

    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % ctx.input_args.port)

    # send message and wait for response
    await socket.send(msg)
    vcmrs.debug(f'Message to NNManager: {msg}')

    # wait for reply
    msg = await socket.recv()
    msg = msg.decode('utf-8')
    vcmrs.log(f'Message received from NNManager: {msg}')
    reply = json.loads(msg, object_hook=lambda d: SimpleNamespace(**d))
    return reply

async def nnmanager_decode_async(
    bitstream_fname,
    recon_fname,
    cvc_intra_fname = None,
    intra_cfg = None, 
    ctx = None):
  '''
  Intra frame decoding

  Command
  -------
    action: 'decode'
    bitstream_fname: input bitstream_fname. If in format <fname:start_idx>, the bitstream starts from the start_idx of the bitstream file
    recon_fname: recon_fname
    cvc_intra_fname: cvc intra frame, the output of intra human adapter
    intra_cfg: decoding parameters
      model_id: 
      picture_width:
      picture_height
  '''
  cmd = _get_decode_cmd(
    bitstream_fname,
    recon_fname, 
    cvc_intra_fname,
    intra_cfg)

  # set request 
  reply = await nnmanager_request_async(cmd, ctx)

  return reply.error

def nnmanager_decode(
    bitstream_fname,
    recon_fname,
    cvc_intra_fname = None,
    intra_cfg = None, 
    ctx=None):
  '''
  Intra frame decoding
  '''
  cmd = _get_decode_cmd(
    bitstream_fname,
    recon_fname, 
    cvc_intra_fname,
    intra_cfg)

  # set request 
  reply = nnmanager_request(cmd, ctx)

  return reply.error


def _get_decode_cmd(
    bitstream_fname,
    recon_fname,
    cvc_intra_fname = None,
    intra_cfg = None):
  cmd = SimpleNamespace()
  cmd.action = 'decode'
  cmd.bitstream_fname = bitstream_fname
  cmd.recon_fname = recon_fname
  cmd.cvc_intra_fname = cvc_intra_fname

  if intra_cfg:
    cmd.intra_cfg = json.dumps(intra_cfg.__dict__)
  else:
    cmd.intra_cfg = None

  return cmd
 

def nnmanager_inter_machine_adapter(
    input_fname, 
    output_fname, 
    ctx,
    gt_fname = None,
    param = None, 
    video_info = None):
  '''
  Apply inter machine adapter

  Command:
    input_fname: input file for inter machine adapter. The data format is in YUV420 10-bit
    output_fname: output file of the inter machine adapter, the data format should be in RGB 8-bit
    gt_fname: uncompressed input frame
    param: parameters for inter machine adapter. 
      qp: frame qp value for the inter machine adapter
    video_info: video information
  '''
  cmd = SimpleNamespace()
  cmd.action = 'inter-machine-adapter'
  cmd.input_fname = input_fname
  cmd.output_fname = output_fname
  cmd.gt_fname = gt_fname

  if param:
    cmd.param = json.dumps(param.__dict__)
  else:
    cmd.param = None

  if video_info:
    cmd.video_info = json.dumps(video_info.__dict__)
  else:
    cmd.video_info = None
 
  # set request 
  reply = nnmanager_request(cmd, ctx)

  return reply


