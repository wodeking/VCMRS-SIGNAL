#!/usr/bin/env python

# process SUF-HW dataset using VTM as the inner codec
# number of tasks 84
#
# Usage of the script: 
#   <script_name> <task_id>
# This script may be useful to process the data on a cluster using CPUs. 
# The script process one item identified by the task_id. The task id is from
# 1 to the total number of tasks. 
# 


import os
import sys
import numpy as np
import time
import glob
import shutil

import sfu_config
import vcmrs

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils



timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'SFU_LD'
output_dir = f'./output/{test_id}'
log_dir=f"{output_dir}/coding_log"

print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')
##############################################################
# configuration
num_workers = 1 # for CPU encoding, one CPU per video

data_dir = '../../Data/SFU'

# check input data
utils.check_roi_descriptor_file('../../Data')


# set test_id and output_dir


# get number of tasks
seq_ids = list(sfu_config.seq_cfg.keys())
num_seqs = len(seq_ids)
num_tasks = num_seqs * len(sfu_config.seq_cfg[seq_ids[0]])

print("number of tasks: ", num_tasks)

task_id = int(sys.argv[1]) - 1

# init environment
ini_path = os.path.join('output', 'working_dir', 'SFU_LD_ini')
os.makedirs(ini_path, exist_ok=True)

#####################################################################
# prepare tasks
seq_idx = task_id % num_seqs
qp_idx = task_id // num_seqs
print(f'Total number of task: {num_tasks}, task id: {task_id}, seq_id: {seq_idx}, qp_idx: {qp_idx}')

seq_id = list(sfu_config.seq_dict.keys())[seq_idx]
qp = f'qp{qp_idx}'
seq_class,fname = sfu_config.seq_dict[seq_id]
quality = sfu_config.seq_cfg[seq_id][qp_idx][0]
print(f'processing seq {fname} with {qp}: {quality}')

bitstream_fname = os.path.join('bitstream', fname, f'{fname}_{qp}.bin')
recon_fname = os.path.join('recon', f'{fname}_{qp}')

# process chunks in parallel
width = sfu_config.res_dict[seq_class][0] 
height = sfu_config.res_dict[seq_class][1]
frame_rate = sfu_config.fr_dict[seq_id][1]
intra_period = sfu_config.fr_dict[seq_id][0]
working_dir=os.path.join('output', 'working_dir', test_id, f'{seq_id}_{qp}')

cfg_fname = os.path.join(ini_path, f'{seq_id}_{qp}.ini')
in_fname = os.path.join(data_dir, f'Class{seq_class}', f'{fname}.yuv')

roi_descriptor = os.path.join(os.path.dirname(data_dir), 'roi_descriptors', f'{fname}.txt')

with open(cfg_fname, 'w') as f:
  f.write(f'''[{in_fname}]
      SourceWidth = {width}
      SourceHeight = {height}
      FrameRate = {frame_rate}
      quality =  {quality}
      output_dir = {output_dir}
      working_dir = {working_dir}
      output_bitstream_fname = {bitstream_fname}
      output_recon_fname = {recon_fname}
      FramesToBeEncoded = {sfu_config.fr_dict[seq_id][2]}
      FrameSkip = {sfu_config.fr_dict[seq_id][3]}
      RoIDescriptor = {roi_descriptor}
      RoIGenerationNetwork = {sfu_config.seq_roi_cfg_network}
  ''')
    

# process SFU sequence
cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--InnerCodec', 'VTM',
      '--single_chunk',
      '--num_workers', num_workers,
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--directory_as_video',
      '--Configuration', 'LowDelay',
      '--SpatialDescriptorMode', 'UsingDescriptor',
      '--debug_source_checksum',
      '--logfile', f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.log",
      cfg_fname,
      ]

print(cmd)

################################################  
# encoding
start_time = time.time()
os.system(' '.join(map(str, cmd)))
print('\n\n=================================================')
print(f'Encoding elapse: {time.time()-start_time}')
print('\n\n')

