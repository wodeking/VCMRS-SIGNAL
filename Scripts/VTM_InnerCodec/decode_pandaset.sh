#!/bin/bash

# Decode Pandaset bitstreams

set -ex
test_id='Pandaset_RA'
base_folder="output"
input_dir="$base_folder/$test_id/bitstream"
output_dir="$base_folder/decode/$test_id/recon"
log_dir="$base_folder/decode/$test_id/decoding_log"
tmp_dir="$base_folder/temp/$test_id"

mkdir -p $output_dir

echo 
echo start decoding ...
echo

#
start_time=$(date +%s)

for bs_fname in $input_dir/*; do
  bname=$(basename $bs_fname)
  # get all files, but we only process the first file
  echo processing $bs_fname 
  working_dir=$tmp_dir/$bname
  python -m vcmrs.decoder \
      --InnerCodec VTM \
      --working_dir $working_dir \
      --output_dir $output_dir \
      --debug_source_checksum \
      --logfile $log_dir/"${bname}.log" \
      $bs_fname
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo 
echo  decoding time: $runtime seconds
echo 
echo  Decoding completed

