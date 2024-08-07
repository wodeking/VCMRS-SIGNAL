#!/usr/bin/env python

# process TVD dataset

import os
import numpy as np
import time
import shutil

import utils
import tvd_tracking_config as config
import vcmrs

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'TVD_tracking_AI_{timestamp}'
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"
vcmrs.setup_logger(name="coding", logfile=f"{log_dir}/main.log")
##############################################################

cuda_devices = [0,1,2,3,4,5,6,7]
processes_per_gpu = 2 # number of process running per GPU
cpu_per_gpu_process = 10 # number of cpu process for each GPU process

data_dir = '../../Data/TVD'
# check input data
utils.check_roi_descriptor_file('../../Data')

################################################################
# encoding

ini_path = os.path.join(base_folder, 'TVD_tracking_ini', 'TVD_tracking_AI')
if os.path.isdir(ini_path):
  shutil.rmtree(ini_path)
os.makedirs(ini_path, exist_ok=True)

tasks = []
log_files = []
start_time = time.time()
for video_id in config.seq_cfg.keys():
  base_video = video_id.split('_')[0]
  video_fname = os.path.join(data_dir, f"{base_video}.yuv")
  for qp_idx, qp in enumerate(config.seq_cfg_ai[video_id]):
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
        IntraPeriod = 1
        quality = {qp}
        NNIntraQPOffset = -5
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
    
    logfile = f"{log_dir}/encoding_{video_id}_qp{qp_idx}.log"
    cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--single_chunk',
      '--num_workers', cpu_per_gpu_process,
      '--InputBitDepth', 8,
      '--InputChromaFormat', '420',
      '--directory_as_video',
      '--Configuration', 'AllIntra',
      '--debug_source_checksum',
      '--logfile', logfile,
        cfg_fname
      ]
    vcmrs.log(cmd),
    tasks.append(cmd)
    log_files.append(logfile)

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log(f'error code: {err}')
vcmrs.log(f'Elapse: {time.time() - start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('all done')

#### Collect coding times #######
times = utils.collect_coding_times(log_files, seq_order_list = list(config.seq_cfg.keys()))
vcmrs.log("Coding time by QP")
vcmrs.log(times)
