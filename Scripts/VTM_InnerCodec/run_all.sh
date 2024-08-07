#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vcm


set -ex

echo '''
  encoding 
'''
num_workers=20
total_num_GPUs=4

echo "Start encoding all using $num_workers CPU cores"

# SFU
num_tasks=84  
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_AI.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_RA.py $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_LD.py $task_id
done

# TVD tracking
num_tasks=42
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $gpu_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_AI.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_RA.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_LD.py $task_id 
done

# Pandaset
if false; then
# num_tasks=438 # for class 1-6
num_tasks=216 # for class 1-3
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $gpu_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --configuration RA $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --configuration LD $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --configuration AI $task_id 
done
fi

sem  --wait

echo "Start encoding all using $num_workers CPU cores"
