#!/bin/bash

# test decode a video output PNG format

set -e
enc_dir=$(echo $0 | sed s/test2_decoding/test_encoding/)
fname="./output/$enc_dir/bitstream/*"
python -m vcmrs.decoder \
  --PostFilter IMA \
  --output_dir ./output/$0 \
  --output_video_format PNG \
  --debug \
  --working_dir ./output/working_dir/$0 \
  --SpatialResample "Bypass" \
  $fname

recon_fname="./output/$enc_dir/recon/*"
dec_fname="./output/$0/*"

echo checking encoder and decoder generating matching results
python ./tools/check_psnr.py --f1 $recon_fname --f2 $dec_fname -t 0 # Bit-exact


