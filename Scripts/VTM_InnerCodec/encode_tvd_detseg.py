#!/usr/bin/env python

# Encode TVD image dataset using VTM as the inner codec
# total number of tasks 996
#
# Usage of the script: 
#   <script_name> <task_id>
# This script may be useful to process the data on a cluster using CPUs. 
# The script process one item identified by the task_id. The task id is from
# 1 to the total number of tasks. 
# 


import os
import sys
import glob
import time
import subprocess

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f"TVD_detseg"

print(f'Test ID: {test_id}')
output_dir = f'./output/{test_id}'

log_dir=f"{output_dir}/coding_log"
print('processing TVD_detseg...')
##############################################################
qps = [22, 27, 32, 37, 42, 47]

# use only CPU
num_workers = 1 # This number should match the number of cores a node has



data_dir = '../../Data/tvd_object_detection_dataset'
working_dir = f'./output/working_dir/{test_id}'
os.makedirs(output_dir, exist_ok=True)


################################################################
# get all images
img_fnames = sorted(glob.glob(os.path.join(data_dir, '*.png')))
total_tasks = len(img_fnames) * len(qps)
print('Total number of tasks: ', total_tasks)


#get task id from input arguments
task_id = int(sys.argv[1]) - 1

qp = qps[task_id // len(img_fnames)]
img_fname = img_fnames[task_id % len(img_fnames)]

bitstream_fname = os.path.join('bitstream', f'QP_{qp}', '{bname}.bin')
recon_fname = os.path.join('recon', f'QP_{qp}', '{bname}')

cmd = ['python',
    '-m', 
    'vcmrs.encoder',
    '--debug',
    '--InnerCodec', 'VTM',
    '--output_dir', output_dir,
    '--output_bitstream_fname', bitstream_fname,
    '--output_recon_fname', recon_fname,
    '--quality', qp, 
    '--num_workers', num_workers,
    '--debug_source_checksum',
    '--logfile', f"{log_dir}/qp{qp}_{task_id}.log",
    '--working_dir',  os.path.join(working_dir, str(qp)),
    img_fname
    ]
    
print(cmd)

start_time = time.time()
subprocess.run(map(str, cmd))
print(f'Elapse: {time.time()-start_time}')
print('all done')


