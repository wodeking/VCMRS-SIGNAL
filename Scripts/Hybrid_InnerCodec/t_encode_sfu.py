#!/usr/bin/env python

# Encode SUF-HW dataset, no encoding time report
# this script shuffle the task better and encode faster than encode_sfu_RA.py and encode_sfu_LD.py

import os
import numpy as np
import time
import glob
import re
import shutil
import utils
import sfu_config
import sys
import vcmrs
import random
import argparse


parser = argparse.ArgumentParser(description='encoding tool for SFU')
parser.add_argument('--enc_mode', type=str, default='RA', help='encoding mode, RA, LD or AI')
parser.add_argument('--test_id', type=str, default='', help='test ID')
args = parser.parse_args()


##############################################################
enc_mode = args.enc_mode

timestamp = time.strftime('%Y%m%d_%H%M%S')
if not args.test_id:
  test_id = f'SFU_{enc_mode}_{timestamp}'
else:
  test_id = args.test_id


base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"

# classes to be evaluated
sfu_classes = ['A', 'B', 'C', 'D']

##############################################################
# device configuration
cuda_devices = [0,1,2,3,4,5,6,7]
processes_per_gpu = 1 # number of process running per GPU
# for 24G GPU, 2 processes per GPU may be used
cpu_per_gpu_process = 10 # number of cpu process for each GPU process


# init environment
data_dir = '../../Data/SFU'
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")

# check input data
utils.check_roi_descriptor_file('../../Data')

# set test_id and output_dir
vcmrs.log(f'Test ID: {test_id}')
vcmrs.log(f'Output directory: {output_dir}\n')

enc_mode_dict = {
  'RA': 'RandomAccess',
  'LD': 'LowDelay',
  'AI': 'AllIntra',
}
enc_mode = enc_mode_dict[enc_mode]


ini_path = os.path.join(base_folder, 'SFU_ini', test_id)
if os.path.isdir(ini_path):
  shutil.rmtree(ini_path)
os.makedirs(ini_path, exist_ok=True)

#####################################################################
# prepare tasks

input_files = []
for seq_id, (seq_class, fname) in sfu_config.seq_dict.items():
  if seq_class not in sfu_classes: continue

  if enc_mode == 'AllIntra':
    qps = sfu_config.seq_cfg_ai[seq_id]
  else:
    # ra and ld has the same configuration
    qps = sfu_config.seq_cfg[seq_id]

  for qp_idx, enc_cfg in enumerate(qps):
    # enc_cfg has format (quality, NNIntraQPOffset)
    quality, intra_qp_offset = enc_cfg
    qp = f'qp{qp_idx}'

    bitstream_fname = os.path.join('bitstream', fname, f'{fname}_{qp}.bin')
    recon_fname = os.path.join('recon', f'{fname}_{qp}')

    # process chunks in parallel
    width = sfu_config.res_dict[seq_class][0] 
    height = sfu_config.res_dict[seq_class][1]
    frame_rate = sfu_config.fr_dict[seq_id][1]
    intra_period = 1 if enc_mode=='AllIntra' else sfu_config.fr_dict[seq_id][0]
    working_dir = os.path.join(base_folder, 'working_dir', test_id, f'{seq_id}_{qp}')

    cfg_fname = os.path.join(ini_path, f'{seq_id}_{qp}.ini')
    in_fname = os.path.join(data_dir, f'Class{seq_class}', f'{fname}.yuv')

    roi_descriptor = os.path.join(os.path.dirname(data_dir), 'roi_descriptors', f'{fname}.txt')
    with open(cfg_fname, 'w') as f:
      f.write(f'''[{in_fname}]
        SourceWidth = {width}
        SourceHeight = {height}
        FrameRate = {frame_rate}
        IntraPeriod = {intra_period}
        quality =  {quality}
        NNIntraQPOffset = {intra_qp_offset}
        output_dir = {output_dir}
        working_dir = {working_dir}
        output_bitstream_fname = {bitstream_fname}
        output_recon_fname = {recon_fname}
        FramesToBeEncoded = {sfu_config.fr_dict[seq_id][2]}
        FrameSkip = {sfu_config.fr_dict[seq_id][3]}
        RoIDescriptor = {roi_descriptor}
      ''')
    
    input_files.append(cfg_fname)

# divide the input files into tasks
random.shuffle(input_files)
n_tasks = len(cuda_devices) * processes_per_gpu
input_chunks = np.array_split(input_files, n_tasks)

tasks = []
for idx, chunk in enumerate(input_chunks):
  logfile=f"{log_dir}/encoding_{idx}.log"
  cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--num_workers', cpu_per_gpu_process,
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--directory_as_video',
      '--Configuration', enc_mode,
      '--logfile', logfile] + list(chunk)
  tasks.append(cmd) 

################################################  
# encoding


start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log('\n\n=================================================')
vcmrs.log(f'Encoding error code: {err}')
vcmrs.log(f'Encoding elapse: {time.time()-start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('\n\n')

