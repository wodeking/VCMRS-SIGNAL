#!/bin/bash

# Decode SFU bitstreams

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
tmp_dir="$base_folder/temp/$test_id"
cuda_device=$2
export CUDA_VISIBLE_DEVICES=$cuda_device

mkdir -p "$output_dir"

echo 
echo start decoding ...
echo

#
start_time=$(date +%s)

for fname in "$input_dir"/*; do
  bname=$(basename "$fname")
  for bs_fname in "$fname"/*; do
    video_fname=$(echo "$bname" | sed s/_[^_]*$//g)
    echo processing "$bs_fname" 
    working_dir=$tmp_dir/$bname
    qp=$(basename "$bs_fname" | sed 's/.*\?\(qp[0-9]\+\).*\?[.]bin/\1/')
    python -m vcmrs.decoder \
      --working_dir "$working_dir" \
      --output_dir "$output_dir" \
      --output_recon_fname "${bname}_${qp}" \
      --debug_source_checksum \
      --logfile "$log_dir"/"${video_fname}_${qp}.log" \
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


