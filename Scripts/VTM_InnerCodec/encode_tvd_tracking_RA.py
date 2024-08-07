#!/usr/bin/env python

# process TVD dataset using VTM as the inner codec
# number of tasks 42
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
import tvd_tracking_config as config

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


timestamp = time.strftime('%Y%m%d_%H%M%S')
# set test_id and output_dir
test_id = f'TVD_tracking_RA'
output_dir = f'./output/{test_id}'

log_dir=f"{output_dir}/coding_log"

print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')

##############################################################
# configuration
num_workers = 1 # This number should match the number of cores a node has

data_dir = '../../Data/TVD'

# check input data
utils.check_roi_descriptor_file('../../Data')


################################################################
# preparing environment

# setup ini file path
ini_path = os.path.join('output', 'working_dir', 'TVD_tracking_RA_ini')
os.makedirs(ini_path, exist_ok=True)

task_id = int(sys.argv[1]) - 1
seqs = list(config.seq_cfg.keys())
num_seqs = len(seqs)
num_tasks = num_seqs * len(config.seq_cfg['TVD-01_1'])
seq_idx = task_id % num_seqs
qp_idx = task_id // num_seqs
print(f'Total number of task: {num_tasks}, task id: {task_id}, seq_id: {seq_idx}, qp_idx: {qp_idx}')

video_id = seqs[seq_idx]
qp = config.seq_cfg[video_id][qp_idx]
base_video = video_id.split('_')[0]
video_fname = os.path.join(data_dir, f"{base_video}.yuv")

bitstream_fname = os.path.join('bitstream', f'{video_id}_qp{qp_idx}.bin')
recon_fname = os.path.join('recon', f'qp{qp_idx}', f'{video_id}')
working_dir = f"./output/working_dir/{test_id}/{video_id}_qp{qp_idx}"

roi_descriptor = os.path.join(os.path.dirname(data_dir), 'roi_descriptors', f'{video_id}.txt')

################################################  
# create config file
cfg_fname = os.path.join(ini_path, f'{video_id}_qp{qp_idx}.ini')
with open(cfg_fname, 'w') as f:
  f.write(f'''[{video_fname}]
        SourceWidth = 1920
        SourceHeight = 1080
        FrameRate = 50
        IntraPeriod = 64
        quality = {qp}
        working_dir = {working_dir}
        output_dir = {output_dir}
        output_bitstream_fname = {bitstream_fname}
        output_recon_fname = {recon_fname}
        FramesToBeEncoded = {config.fr_dict[video_id][2]}
        FrameSkip = {config.fr_dict[video_id][3]}
        InputBitDepth = {config.fr_dict[video_id][4]}
        RoIDescriptor = {roi_descriptor}
        RoIGenerationNetwork = {config.seq_roi_cfg_network}
  ''')
  

cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--InnerCodec', 'VTM',
      '--single_chunk',
      '--directory_as_video',
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--Configuration', 'RandomAccess',
      '--SpatialDescriptorMode', 'UsingDescriptor',
      '--num_workers', num_workers,
      '--debug_source_checksum',
      '--logfile', f"{log_dir}/encoding_{video_id}_qp{qp_idx}_{timestamp}.log",
      cfg_fname,
      ] 
        
################################################  
# encoding
start_time = time.time()
print(cmd)
os.system(' '.join(map(str,cmd)))
print(f'Elapse: {time.time()-start_time}')
print('all done')





