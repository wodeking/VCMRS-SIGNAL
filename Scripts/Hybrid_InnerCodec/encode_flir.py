#!/usr/bin/env python

# process TVD dataset

import os
import shutil
import time

import utils
import vcmrs
timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'flir_{timestamp}'
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
log_dir =  f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")
working_dir = f'./{base_folder}/working_dir/{test_id}'
##############################################################
qps = [22, 27, 32, 37, 42, 47]
cuda_devices = [0,1,2,3,4,5] # GPUs for this dataset
processes_per_gpu = 1
vcmrs.log(f'Test ID: {test_id}')
os.makedirs(output_dir, exist_ok=True)

################################################################

data_dir = '../../Data/FLIR_MPEG135/thermal_images'
tasks = []
for qp in qps:
  bitstream_fname = os.path.join('bitstream', f'QP_{qp}', '{bname}.bin')
  recon_fname = os.path.join('recon', f'QP_{qp}', '{bname}')

  cmd = ['python',
    '-m', 'vcmrs.encoder',
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
    
start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)

vcmrs.log(f'error code {err}')
vcmrs.log(f'Elapse: {time.time()-start_time}, GPUs {len(cuda_devices)}')

### Collect coding time ####
times = utils.collect_coding_times([f"{log_dir}/qp{qp}.log" for qp in qps], file_regx="(.*)(qp\d+).*")
vcmrs.log("Coding time by QP:")
vcmrs.log(times)
