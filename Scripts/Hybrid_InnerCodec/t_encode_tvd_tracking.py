#!/usr/bin/env python

# encode TVD dataset in parallel
# this script shuffle the task better and encode faster than encode_tvd_tracking_RA.py and encode_tvd_tracking_LD.py

import os
import numpy as np
import time
import shutil

import utils
import tvd_tracking_config as config
import vcmrs
import random
import argparse

parser = argparse.ArgumentParser(description='encoding tool for TVD')
parser.add_argument('--enc_mode', type=str, default='RA', help='encoding mode, RA, LD or AI')
parser.add_argument('--test_id', type=str, default='', help='test ID')
args = parser.parse_args()


##############################################################
enc_mode = args.enc_mode

timestamp = time.strftime('%Y%m%d_%H%M%S')
if not args.test_id:
  test_id = f'TVD_{enc_mode}_{timestamp}'
else:
  test_id = args.test_id



base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"

cuda_devices = [0,1,2,3,4,5,6,7]
processes_per_gpu = 1 # number of process running per GPU
cpu_per_gpu_process = 10 # number of cpu process for each GPU process



##############################################################
# environment setup
data_dir = '../../Data/TVD'
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

# check input video md5sum
print('checking input video md5sum...')
for video_id, md5sum in config.md5sum.items():
  utils.check_md5sum(os.path.join(data_dir, f"{video_id}.yuv"), md5sum)


################################################################
# encoding

ini_path = os.path.join(base_folder, 'TVD_tracking_ini', test_id)
if os.path.isdir(ini_path):
  shutil.rmtree(ini_path)
os.makedirs(ini_path, exist_ok=True)

input_files = []
for video_id in config.seq_cfg.keys():
  base_video = video_id.split('_')[0]
  video_fname = os.path.join(data_dir, f"{base_video}.yuv")

  if enc_mode == 'AllIntra':
    qps = config.seq_cfg_ai[video_id]
  else:
    # ra and ld has the same configuration
    qps = config.seq_cfg[video_id]

  intra_period = 1 if enc_mode=='AllIntra' else config.fr_dict[video_id][0]

  for qp_idx, qp in enumerate(qps):
    bitstream_fname = os.path.join('bitstream', f'{video_id}_qp{qp_idx}.bin')
    recon_fname = os.path.join('recon', f'qp{qp_idx}', f'{video_id}')
    output_frame_format = "{frame_idx:06d}.png"
    working_dir = f"{base_folder}/working_dir/{test_id}/{video_id}_qp{qp_idx}"

    roi_descriptor = os.path.join(os.path.dirname(data_dir), 'roi_descriptors', f'{video_id}.txt')

    # create config file
    cfg_fname = os.path.join(ini_path, f'{video_id}_qp{qp_idx}.ini')
    with open(cfg_fname, 'w') as f:
      f.write(f'''[{video_fname}]
        SourceWidth = 1920
        SourceHeight = 1080
        FrameRate = 50
        IntraPeriod = {intra_period}
        quality = {qp}
        NNIntraQPOffset = -5
        working_dir = {working_dir}
        output_dir = {output_dir}
        output_bitstream_fname = {bitstream_fname}
        output_recon_fname = {recon_fname}
        FramesToBeEncoded = {config.fr_dict[video_id][2]}
        FrameSkip = {config.fr_dict[video_id][3]}
        InputBitDepth = {config.fr_dict[video_id][4]}
        InputChromaFormat = 420
        RoIDescriptor = {roi_descriptor}
      ''')
    input_files.append(cfg_fname)
    
# divide the input files into tasks
random.shuffle(input_files)
n_tasks = len(cuda_devices) * processes_per_gpu
input_chunks = np.array_split(input_files, n_tasks)

tasks = []
for idx, chunk in enumerate(input_chunks):
  logfile = f"{log_dir}/encoding_{idx}.log"
  cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--num_workers', cpu_per_gpu_process,
      '--directory_as_video',
      '--Configuration', enc_mode,
      '--logfile', logfile,
      ] + list(chunk)
  tasks.append(cmd)

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log(f'error code: {err}')
vcmrs.log(f'Elapse: {time.time() - start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('all done')


