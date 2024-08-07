#!/bin/bash

# test decoding a video  in YUV format

set -e
enc_dir=$(echo $0 | sed s/test2_decoding/test_encoding/)
fname="./output/$enc_dir/bitstream/*"
python -m vcmrs.decoder \
  --output_dir ./output/$0 \
  --debug \
  --working_dir ./output/working_dir/$0 \
  --TemporalResample 'Bypass' \
  --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname

recon_fname="./output/$enc_dir/recon/*"
dec_fname="./output/$0/*"

echo checking encoder and decoder generating matching results
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $dec_fname --f2_bitdepth 10 -t 0 # Bit-exact


