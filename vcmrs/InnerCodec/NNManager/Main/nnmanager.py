# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
# NNManager

import hashlib
import os
import time
import zmq
import json
from types import SimpleNamespace
import argparse
import cv2
from pathlib import Path
import torch
import importlib
import vcmrs
from vcmrs.InnerCodec.NNManager.Main.cache_manager import empty_cache_memory

from vcmrs.Utils import utils
from vcmrs.Utils import data_utils
from vcmrs.Utils import io_utils
from vcmrs.InnerCodec.NNManager.Main.inter_machine_adapter_controller import InterMachineAdapterController
from vcmrs.InnerCodec.NNManager.Main.intra_human_adapter_controller import IntraHumanAdapterController

import e2evc.Utils as e2eutils


# Some hardcoded values to be changed
check_md5 = False

def get_arguments():
  parser = argparse.ArgumentParser()

  # arguments for process subroutine
  # data I/O
  parser.add_argument('--port', type=str, default='*', 
                      help='port number for communication. Default: * indicate the server will find a free port and write the port number to info_file in format: port=xxxx')

  parser.add_argument('--info_file', type=str, default='',
                      help='report the port number of the ZMQ server')
  
  parser.add_argument('--logfile', default='', type=str, help='Path of the file where the logs are saved. By default the logs are not saved.')
  parser.add_argument('--debug', action='store_true', 
                      help='In debug mode, more logs are printed, intermediate files are not removed')

  parser.add_argument('--IntraCodec', type=str, default="LIC",
                      help='Intra codec name')

  parser.add_argument('--NNConfigs', type=str, default="run_config.yaml",
                      help='Intra codec config file name. Default: run_config.yaml')

  args = parser.parse_args()

  return args


def init(args):
  vcmrs.log('NNManager: initializing ...')
  utils.make_deterministic(True)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # args.model_fname = os.path.join(codec_dir, 'NNManager', 'Pretrained', "intra_codec", "model_200.pth.tar")
  if not os.path.isabs(args.NNConfigs):
    args.NNConfigs = os.path.join(os.path.dirname(__file__), '..', 'Pretrained', args.NNConfigs)
  args.model_configs = io_utils.read_yaml(configs_path=args.NNConfigs)
  #IC_controller = IntraCodecController(IHA_controller, args, device=device, max_preloaded=5)
  IC_controller = load_intra_codec_controller(args, device=device)

  IHA_controller = IntraHumanAdapterController(args, device=device, max_preloaded=5)
  IMA_controller = InterMachineAdapterController(args, device=device, max_preloaded=5)
  return {
    "IHA_controller": IHA_controller,
    "IC_controller": IC_controller,
    "IMA_controller": IMA_controller,
  }

def load_intra_codec_controller(args, device):
  ic_name = args.IntraCodec
  ic_fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'NNIntraCodec', ic_name, 'intra_codec_controller.py')
  spec = importlib.util.spec_from_file_location(f"vcmrs.InnerCodec.NNManager.NNIntraCodec.{ic_name}", ic_fname)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  ic_controller = foo.IntraCodecController(args, device=device)
  return ic_controller 


def process_YUV(req, intra_codec_controller): 
  # TODO: provide request interface for this kind of YUV splitted frame processing
  intra_cfg = req.intra_cfg
  yuv_fname, frame_idx = req.input_fname.split(':')
  frame_idx = int(frame_idx)

  H,W,C = req.video_info.resolution
  img_yuv = data_utils.get_frame_from_raw_video(yuv_fname, frame_idx, 
    W=W, H=H, 
    chroma_format=req.video_info.chroma_format,
    bit_depth=req.video_info.bit_depth
    )

  # recon img
  img_bgr = data_utils.cvt_yuv444p10b_to_bgr(img_yuv)
  rgb_input_fp = Path(req.recon_fname).parent.joinpath(Path(yuv_fname).stem + "RBG.png")
  #rgb_output_fp = os.path.splitext(req.recon_fname)[0] + "RBG.png"
  rgb_output_fp = req.recon_fname
  cv2.imwrite(rgb_input_fp.as_posix(), img_bgr)
  vcmrs.debug(f"Converted intra input to RGB at {rgb_input_fp}")
  rets = intra_codec_controller.code_image(input_fp=rgb_input_fp, 
    output_bitstream_fp=req.bitstream_fname,
    output_image_fp=rgb_output_fp, 
    intra_cfg = req.intra_cfg
  )

  vcmrs.log(f"Intra codec error code: {rets['error']}")
  
  return rets

def process_PNG(req, intra_codec_controller):
  #handle png file
  intra_cfg = req.intra_cfg
  intra_cfg.req_qp = req.intra_cfg.video_qp
  rets = intra_codec_controller.code_image(
    input_fp=req.input_fname, 
    output_bitstream_fp=req.bitstream_fname,
    output_image_fp=req.recon_fname, 
    intra_cfg=intra_cfg
  )
  vcmrs.log(f"Intra codec error code: {rets['error']}")
  if check_md5:
    rets["intra_checksum"] = hashlib.md5(cv2.imread(req.recon_fname)).hexdigest()
    vcmrs.log(f'Intra coding hash: {rets["intra_checksum"]}')
  
  return rets
  
def apply_iha(req, frame_qp, intra_human_apdapter_controller):
  # apply intra human adapter
  input_fname = req.recon_fname
  o_adapted_fp = req.cvc_intra_fname
  IHA_flag = req.intra_cfg.IHA_flag

  rets = {'error': 0, 'model_id': 0}
  if o_adapted_fp:
    if IHA_flag==1:
      s_time = time.time()
      rets = intra_human_apdapter_controller.apply_intra_human_adapter(
        input_image=input_fname, 
        frame_qp=frame_qp)
      adapted_img = rets['values']['adapted_image'] # CHW [0, 1]
      vcmrs.log(f"{req.intra_cfg.time_tag} IHA done. Time = {(time.time() - s_time):.6}(s)")
    else:
      # simply convert to YUV
      adapted_img = e2eutils.load_img_tensor(input_fname).squeeze(0)

    e2eutils.save_rgb2yuv420p_10b_precise(adapted_img, o_adapted_fp)

  return rets

def encode(req, intra_codec_controller, intra_human_adapter_controller):
  vcmrs.log(f'Processing {req.input_fname}...')
  rep = SimpleNamespace()

  # validate input
  if req.intra_cfg.IHA_flag: 
    assert req.cvc_intra_fname, 'cvc_intra_fname must be given if IHA_flag is 1'

  # simulate processing
  # ideally the NNManager should check if the intra frame is already processed. if yes, 
  # it shall not process it again.
  s_time = time.time()
  if ':' in req.input_fname:
    # handel YUV file: yuv_file.yuv:idx
    rets = process_YUV(req, intra_codec_controller)
  else:
    rets = process_PNG(req, intra_codec_controller)

  vcmrs.log(f"{req.intra_cfg.time_tag} LIC encoding done. Time = {(time.time() - s_time):.6}(s)")

  if rets['error'] == 0:
    rep.model_id = rets['values']['model_id']
    frame_qp = rets['values']['frame_qp']
    # apply IHA
    rets = apply_iha(req, frame_qp, intra_human_adapter_controller)

  rep.error = rets['error']

  vcmrs.log("----------------------------------")
  return rep
 
def decode(req, intra_codec_controller, intra_human_adapter_controller):
  rep = SimpleNamespace()
  s_time = time.time()
  rets = intra_codec_controller.decode_bitstream(
      input_fp=req.bitstream_fname, 
      output_image_fp=req.recon_fname, 
      intra_cfg = req.intra_cfg
  )
  vcmrs.log(f"{req.intra_cfg.time_tag} LIC decoding done. Time = {(time.time() - s_time):.6}(s)")
  s_time = time.time()
  if rets['error'] == 0:
    rep.model_id = rets['values']['model_id']

    # apply IHA
    # set frame QP for intra frame, required by IHA
    frame_qp = rets['values']['frame_qp']

    rets = apply_iha(req, frame_qp, intra_human_adapter_controller)
    vcmrs.log(f"{req.intra_cfg.time_tag} IHA done. Time = {(time.time() - s_time):.6}(s)")

  rep.error = rets['error']
  if check_md5:
    rep.checksum = hashlib.md5(cv2.imread(req.recon_fname)).hexdigest()
  return rep
 
def inter_machine_adapter(req, inter_machine_adapter_controller):
  s_time = time.time()
  input_fname = req.input_fname
  output_fname = req.output_fname
  gt_fname = req.gt_fname # may be None
  video_info = req.video_info   
  param = req.param
  rets = inter_machine_adapter_controller.apply_inter_machine_adapter(input_fp=input_fname, output_image_fp=output_fname, video_info=video_info, param=param)
 
  rep = SimpleNamespace()
  error_code = rets['error']
  if error_code == 0:
   rep.model_id = rets['values']['model_id']
  rep.error = error_code
  time_tag = getattr(param, "item_tag", "") 
  vcmrs.log(f"{time_tag} IMA done. Time = {(time.time() - s_time):.6}(s)")
  return rep
 
def main(args):
  # args = get_arguments()
  global log
  vcmrs.setup_logger(name="nnmanager", logfile=args.logfile, debug=args.debug)
  rets = init(args)
  intra_codec_controller = rets["IC_controller"]
  intra_human_adapter_controller = rets["IHA_controller"]
  inter_machine_adapter_controller = rets["IMA_controller"]

  # start message loop
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind("tcp://*:%s" % args.port)
  # get port number
  port = socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii').split(':')[-1]
  if args.info_file:
    info = {'port': port}
    with open(args.info_file, 'w') as f:
      json.dump(info, f)
  vcmrs.log(f'Socket connected to port {port}')
  
  while True:
      # receive message
      vcmrs.log('Ready for next request')
      msg = socket.recv().decode('utf-8')
      req = json.loads(msg, object_hook=lambda d: SimpleNamespace(**d))
      if getattr(req, "intra_cfg", False):
         req.intra_cfg = json.loads(req.intra_cfg, object_hook=lambda d: SimpleNamespace(**d))
      if getattr(req, "video_info", False):
         req.video_info = json.loads(req.video_info, object_hook=lambda d: SimpleNamespace(**d))
      if getattr(req, "param", False):
         req.param = json.loads(req.param, object_hook=lambda d: SimpleNamespace(**d))
      
      vcmrs.log(f"Received request for action: {req.action}")
      vcmrs.debug(f"Request message: {req}")

      if req.action == 'encode':
        req.intra_cfg.time_tag = getattr(req.intra_cfg, "item_tag", "")
        rep = encode(req, intra_codec_controller, intra_human_adapter_controller)
      elif req.action == 'decode':
        req.intra_cfg.time_tag = getattr(req.intra_cfg, "item_tag", "") 
        rep = decode(req, intra_codec_controller, intra_human_adapter_controller)
      elif req.action == 'inter-machine-adapter':
        rep = inter_machine_adapter(req, inter_machine_adapter_controller)
      else:
        assert False, f"unsupported action: {req.action}"
      
      msg = json.dumps(rep.__dict__).encode()
      socket.send(msg)

      # Todo: explicitly clear memory, otherwise GPU OOM
      empty_cache_memory()

if __name__ == '__main__':
  args = get_arguments()

  main(args)
