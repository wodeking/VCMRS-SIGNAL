#!/usr/bin/env python

# Encode TVD image dataset

import os
import numpy as np
import time
import shutil
import utils
import vcmrs

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f"TVD_detseg_{timestamp}"
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
working_dir = f'{base_folder}/working_dir/{test_id}'
log_dir =  f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")
##############################################################
qps = [22, 27, 32, 37, 42, 47]
cuda_devices = [0,1,2,3,4,5,6,7] # GPU devices
processes_per_gpu = 1 # 16G memeory


vcmrs.log('processing TVD_detseg...')
vcmrs.log(f'Test ID: {test_id}')

data_dir = '../../Data/tvd_object_detection_dataset'
os.makedirs(output_dir, exist_ok=True)


################################################################

tasks = []
task_name = []
for qp in qps:
  bitstream_fname = os.path.join('bitstream', f'QP_{qp}', '{bname}.bin')
  recon_fname = os.path.join('recon', f'QP_{qp}', '{bname}')

  cmd = ['python',
    '-m',
    'vcmrs.encoder',
    '--output_dir', output_dir,
    '--output_bitstream_fname', bitstream_fname,
    '--output_recon_fname', recon_fname,
    '--quality', qp,
    '--logfile', f"{log_dir}/qp{qp}.log",
    '--debug_source_checksum',
    '--working_dir',  os.path.join(working_dir, f"QP{qp}"),
    # '--debug',
    data_dir
    ]
  tasks.append(cmd)
  task_name.append(f"QP{qp}")

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
err_dict = {k:v for k,v in zip(task_name, err)}
vcmrs.log(f'error code: {err_dict}')
vcmrs.log(f'total elapse: {time.time() - start_time}')
vcmrs.log('all process has finished')

### Collect coding time ####
times = utils.collect_coding_times([f"{log_dir}/qp{qp}.log" for qp in qps], file_regx="(.*?)(qp\d+).*")
vcmrs.log("Coding time by QP:")
vcmrs.log(times)
