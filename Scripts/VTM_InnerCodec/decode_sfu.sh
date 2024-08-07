#!/bin/bash

# Decode SFU bitstreams

set -ex
test_id='SFU'
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

for fname in $input_dir/*; do
  bname=$(basename $fname)
  # get all files, but we only process the first file
  for bs_fname in $fname/*; do
    video_fname=$(echo $bname | sed s/_[^_]*$//g)
    echo processing $bs_fname 
    working_dir=$tmp_dir/$bname
    qp=$(basename $bs_fname | sed 's/.*\?\(qp[0-9]\+\).*\?[.]bin/\1/')
    python -m vcmrs.decoder \
      --InnerCodec VTM \
      --working_dir $working_dir \
      --output_dir $output_dir \
      --output_recon_fname "${bname}_${qp}" \
      --debug_source_checksum \
      --logfile $log_dir/"${video_fname}_${qp}.log" \
      $bs_fname
  done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo 
echo  decoding time: $runtime seconds
echo 
echo  Decoding completed


