#!/usr/bin/env python

# process SUF-HW dataset 
# number of tasks 102

import os
import sys
import numpy as np
import time
import glob
import shutil
import filecmp
import argparse
import atexit
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandaset_config
import vcmrs


##############################################################
# intput arguments
parser = argparse.ArgumentParser(description='encode Pandaset')

parser.add_argument('--test_id', type=str, default=None, 
  help='Test ID, Default: output.')
parser.add_argument('--configuration', type=str, default='RA', 
  help='configuration, RA/LD/AI, Default: RA')
parser.add_argument('--seq_id', type=str, default=None, 
  help='sequence id, default: None')
parser.add_argument('--seq_qp', type=str, default=None, 
  help='sequence qp, default: None')
parser.add_argument('--classes', type=str, default="1,2,3", 
  help='classes, default: "1,2,3"')
parser.add_argument('--task_id', type=int, default=1, 
  help='Task ID, starting from 1, Default: 0.')

args = parser.parse_args()

assert args.task_id>=1, 'task_id shall start from 1'

if args.configuration == 'RA':
  cfg = 'RandomAccess'
  seq_cfg = pandaset_config.seq_cfg
  num_workers = 1 # 2 intra period for RA mode 
elif args.configuration == 'LD':
  cfg = 'LowDelay'
  seq_cfg = pandaset_config.seq_cfg
  num_workers = 1 # 
elif args.configuration == 'AI':
  cfg = 'AllIntra'
  seq_cfg = pandaset_config.seq_cfg_ai
  num_workers = 1 # 
else:
  print('Unknow configuration: ', args.configuration)
  sys.exit(1)



timestamp = time.strftime('%Y%m%d_%H%M%S')
log_dir=f"coding_log/pandaset_{timestamp}"


##############################################################
# configuration
data_dir = '../../Data/Pandaset_YUV'

# set test_id and output_dir
test_id = args.test_id
if test_id is None: test_id = f'Pandaset_{args.configuration}'

output_dir = f'./output/{test_id}'
print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')

##############################################################
# CTC config

tasks = []
if args.seq_id is None: 
  # get all seq_ids for the class to be processed
  seq_ids_in_classes = []
  for cls in args.classes.split(','):
    seq_ids_in_classes += pandaset_config.cls_dict[str(cls)]

  # encode whole dataset
  for seq_id, (dataset, seq_name) in pandaset_config.seq_dict.items():
    if not int(seq_id) in seq_ids_in_classes: continue 
    for qp_idx,(qp, nn_intra_offset) in enumerate(seq_cfg[seq_id]):
      tasks.append((seq_id, qp_idx, qp, cfg))
else:
  seq_id = args.seq_id
  tasks.append((seq_id, -1, args.seq_qp, cfg))
  

###############################################################

print(f'Total number of task: {len(tasks)}')

# get number of tasks
task_id = args.task_id - 1
seq_id, qp_idx, qp, cfg = tasks[task_id]

print('Tasks: ')
print(*list(zip(range(len(tasks)), tasks)), sep='\n')

print(f'Encoding seq: {seq_id}, qp_idx: {qp_idx}, qp: {qp}, cfg: {cfg}')

seq_class,fname = pandaset_config.seq_dict[seq_id]
if qp_idx == -1:
  bitstream_fname = os.path.join('bitstream', f'{fname}_{qp}.bin')
  recon_fname = os.path.join('recon', f'{fname}_{qp}')
else:
  bitstream_fname = os.path.join('bitstream', f'{fname}_qp{qp_idx}.bin')
  recon_fname = os.path.join('recon', f'{fname}_qp{qp_idx}')

roi_descriptor = os.path.join(os.path.dirname(data_dir), 'roi_descriptors_pandaset', f'{fname}.txt')
spatial_descriptor = os.path.join(os.path.dirname(data_dir), 'spatial_descriptors_pandaset', f'{fname}.csv')

# get seqence information
cls = pandaset_config.seq_dict[seq_id][0]
width = 1920
height = 1080
bit_depth = 8
frame_rate = pandaset_config.fr_dict[seq_id][1]
intra_period = pandaset_config.fr_dict[seq_id][0]

working_dir = os.path.join('output', 'working_dir', test_id, f'{seq_id}_{qp}')

input_fname = os.path.join(data_dir, f'{fname}.yuv')
cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--num_workers', num_workers,
      '--directory_as_video',
      '--logfile', f"{log_dir}/pandaset_{seq_id}_{qp}.log",
      '--output_dir', output_dir,
      '--working_dir', working_dir,
      '--output_bitstream_fname', bitstream_fname,
      '--output_recon_fname', recon_fname,
      '--SourceWidth', width,
      '--SourceHeight', height,
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--FramesToBeEncoded', pandaset_config.fr_dict[seq_id][2],
      '--FrameSkip', pandaset_config.fr_dict[seq_id][3],
      '--FrameRate', frame_rate,
      '--single_chunk',
      '--Configuration', cfg,
      '--IntraPeriod', intra_period,
      '--quality',  qp,
      '--NNIntraQPOffset',  nn_intra_offset,
      '--RoIDescriptor', roi_descriptor,
      '--SpatialDescriptorMode', 'UsingDescriptor',
      '--SpatialDescriptor', spatial_descriptor,
      '--debug_source_checksum',
      input_fname
      ]

print(cmd)

################################################  
# encoding
start_time = time.time()
os.system(' '.join(map(str, cmd)))

print('\n\n=================================================')
print(f'Encoding elapse: {time.time()-start_time}')
print('\n\n')




