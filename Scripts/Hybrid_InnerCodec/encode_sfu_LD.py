#!/usr/bin/env python

# process SUF-HW dataset

import os
from pathlib import Path
import re
import numpy as np
import time
import glob

import shutil

import pandas as pd
import utils
import sfu_config
import sys
import vcmrs

base_folder = "output"
timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'SFU_LD_{timestamp}'
output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")

##############################################################
# configuration
cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7] 
processes_per_gpu = 1 # number of process running per GPU
# for 24G GPU, 2 processes per GPU may be used
# each GPU process for a class
cpu_per_gpu_process = 10 # number of cpu process for each GPU process

# classes to be evaluated
sfu_classes = ['A', 'B','C', 'D']

data_dir = '../../Data/SFU'

vcmrs.log=print
# set test_id and output_dir
vcmrs.log(f'Test ID: {test_id}')
vcmrs.log(f'Output directory: {output_dir}\n')

# check input data
utils.check_roi_descriptor_file('../../Data')


# init environment
ini_path = os.path.join(base_folder, 'SFU_ini', 'SFU_LD')
if os.path.isdir(ini_path):
  shutil.rmtree(ini_path)
os.makedirs(ini_path, exist_ok=True)

#####################################################################
# prepare tasks
tasks = []
log_files = []
for seq_id in sfu_config.seq_cfg.keys():
  if not (seq_id in sfu_config.seq_dict.keys()): continue

  seq_class, fname = sfu_config.seq_dict[seq_id]
  if seq_class not in sfu_classes: continue
  for qp_idx, enc_cfg in enumerate(sfu_config.seq_cfg[seq_id]):
    # enc_cfg has format (quality, NNIntraQPOffset)
    quality, intra_qp_offset = enc_cfg
    qp = f'qp{qp_idx}'

    bitstream_fname = os.path.join('bitstream', fname, f'{fname}_{qp}.bin')
    recon_fname = os.path.join('recon', f'{fname}_{qp}')

    # process chunks in parallel
    width = sfu_config.res_dict[seq_class][0] 
    height = sfu_config.res_dict[seq_class][1]
    frame_rate = sfu_config.fr_dict[seq_id][1]
    intra_period = sfu_config.fr_dict[seq_id][0]
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
        RoIGenerationNetwork = {sfu_config.seq_roi_cfg_network}
      ''')

    logfile=f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.log"
    cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--single_chunk',
      '--num_workers', cpu_per_gpu_process,
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--directory_as_video',
      '--Configuration', 'LowDelay',
      '--debug_source_checksum',
      '--logfile', logfile,
      cfg_fname
      ]
    
    log_files.append(logfile)
    vcmrs.log(cmd)
    tasks.append(cmd) 

###############################################  
# encoding
start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log('\n\n=================================================')
vcmrs.log(f'Encoding error code: {err}')
vcmrs.log(f'Encoding elapse: {time.time()-start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('\n\n')

#### Collect coding times #######
times = utils.collect_coding_times(log_files, seq_order_list = list(sfu_config.seq_dict.keys()))
vcmrs.log("Coding time by QP:")
vcmrs.log(times)



