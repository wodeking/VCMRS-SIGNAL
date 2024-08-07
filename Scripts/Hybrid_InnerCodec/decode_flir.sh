#!/bin/bash

# Decode FLIR bitstreams

set -e

test_id='FLIR'
base_folder="output"
input_dir="$base_folder/$test_id/bitstream"
output_dir="$base_folder/decode/$test_id/recon"
log_dir="$base_folder/decode/$test_id/decoding_log"

mkdir -p $output_dir

start_time=$(date +%s)
dec_exe='-m vcmrs.decoder'

echo 
echo start decoding ...
echo 

# decoding, one QP per GPU
qp_values=("22" "27" "32" "37" "42" "47")
for i in "${!qp_values[@]}"; do
    CUDA_VISIBLE_DEVICES=$i python $dec_exe --output_dir $output_dir/QP_${qp_values[i]} $input_dir/QP_${qp_values[i]} --debug_source_checksum --logfile $log_dir/qp${qp_values[i]}.log &
done

# decoding using 2 GPUs
# CUDA_VISIBLE_DEVICES=0 python $dec_exe --output_dir $output_dir/QP_22 $input_dir/QP_22 && \
# CUDA_VISIBLE_DEVICES=0 python $dec_exe --output_dir $output_dir/QP_27 $input_dir/QP_27 && \
# CUDA_VISIBLE_DEVICES=0 python $dec_exe --output_dir $output_dir/QP_32 $input_dir/QP_32 &

# CUDA_VISIBLE_DEVICES=1 python $dec_exe --output_dir $output_dir/QP_37 $input_dir/QP_37 && \
# CUDA_VISIBLE_DEVICES=1 python $dec_exe --output_dir $output_dir/QP_42 $input_dir/QP_42 && \
# CUDA_VISIBLE_DEVICES=1 python $dec_exe --output_dir $output_dir/QP_47 $input_dir/QP_47 &

# wait until all decoding completed
wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo 
echo  decoding time: $runtime seconds
echo 
echo  Decoding completed


