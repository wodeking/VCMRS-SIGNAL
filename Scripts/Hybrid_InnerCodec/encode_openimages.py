#!/usr/bin/env python

# process open images

import os
import shutil
import numpy as np
import time

import utils
import vcmrs

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'openimages_{timestamp}'
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
working_dir = f'{base_folder}/working_dir/{test_id}'
log_dir = f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")
##############################################################
qps = [22, 27, 32, 37, 42, 47]
cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7] # CUDA devices
processes_per_gpu = 1 # number of processes running per GPU
n_chunks = 6 # split images into chunks, so they can be parallelized better

vcmrs.log('Encoding OpenImages...')
vcmrs.log(f'Test ID: {test_id}')

data_dir = '../../Data/OpenImages/validation/'

# prepare working dir
os.makedirs(working_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

################################################################
# get input images
with open('openimages_input.lst', 'r') as f:
  input_fnames = f.read().splitlines()

# split input images into chuncks
chunks = np.array_split(input_fnames, n_chunks)
chunk_fnames = []
for idx in range(n_chunks):
  chunk_fname = os.path.join(working_dir, f'openimages_{idx}.ini')
  with open(chunk_fname, 'w') as of:
      of.write('['+' '.join(chunks[idx])+']')
      of.write(f'\ninput_prefix = {data_dir}')
  chunk_fnames.append(chunk_fname)

# prepare tasks
tasks = []
task_name = []
for qp in qps:
  bitstream_fname = os.path.join('bitstream', f'QP_{qp}', '{bname}.bin')
  recon_fname = os.path.join('recon', f'QP_{qp}', '{bname}')

  for chunk_idx,fname in enumerate(chunk_fnames):
    cmd = ['python',
      '-m', 'vcmrs.encoder',
      '--output_dir', output_dir,
      '--output_bitstream_fname', bitstream_fname,
      '--output_recon_fname', recon_fname,
      '--working_dir',  os.path.join(working_dir, f"QP{qp}_{chunk_idx}"),
      '--quality', qp, 
      '--debug_source_checksum',
      '--logfile', f"{log_dir}/qp{qp}.log",
      # '--debug',
      fname
      ]
    tasks.append(cmd)
    task_name.append(f"QP{qp}_{chunk_idx}")

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
err_dict = {k:v for k,v in zip(task_name, err)}
vcmrs.log(f'error code: {err_dict}')
vcmrs.log(f'total elapse: {time.time() - start_time}')
vcmrs.log('all process has finished')

### Collect coding time ####
times = utils.collect_coding_times([f"{log_dir}/qp{qp}.log" for qp in qps], file_regx="(.*?)(qp\d+).*")
# times['encoding_time']=times['encoding_time']/8189*5000
vcmrs.log("Coding time by QP on 5000 images:")
vcmrs.log(times)
times.to_csv(os.path.join(log_dir,"codingtime_estimated.csv"))
