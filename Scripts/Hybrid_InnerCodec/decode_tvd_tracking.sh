#!/bin/bash

# Decode TVD tracking bitstreams

set -ex

test_id=$1
base_folder="output"

pushd $base_folder
test_id=( "$test_id" ) # Resolves wildcards
test_id=$(echo $test_id)
cd "$test_id" || exit # Exit if folder does not exist
popd

input_dir="$base_folder/$test_id/bitstream"
output_dir="$base_folder/$test_id/decode/recon"
log_dir="$base_folder/$test_id/decode/decoding_log"
qps='qp0 qp1 qp2 qp3 qp4 qp5'

cuda_device=$2
export CUDA_VISIBLE_DEVICES=$cuda_device

mkdir -p "$output_dir"

video_ids='TVD-01_1 TVD-01_2 TVD-01_3 TVD-02_1 TVD-03_1 TVD-03_2 TVD-03_3'

#
start_time=$(date +%s)
for video_id in $video_ids; do
  for qp in $qps; do
    bs_fname=$input_dir/${video_id}_${qp}.bin
    echo processing "$bs_fname" ...
    python -m vcmrs.decoder \
      --working_dir "$base_folder/dec_temp/$test_id" \
      --output_dir "$output_dir"/"$qp" \
      --output_recon_fname "$video_id" \
      --debug_source_checksum \
      --logfile "${log_dir}/${video_id}_${qp}.log" \
      "$bs_fname"
  done
done

python collect_coding_time.py "$log_dir" decoding

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo 
echo  decoding time: $runtime seconds
echo 
echo  Decoding completed


