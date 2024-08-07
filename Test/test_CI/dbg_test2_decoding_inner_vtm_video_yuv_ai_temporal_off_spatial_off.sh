#!/bin/bash

# test decoding an video encoded in AI configuration

set -e
enc_dir=$(echo $0 | sed s/test2_decoding/test_encoding/)
fname="./output/$enc_dir/bitstream/*"
python -m vcmrs.decoder \
  --output_dir ./output/$0 \
  --debug \
  --InnerCodec VTM \
  --working_dir ./output/working_dir/$0 \
  --TemporalResample "Bypass" \
  --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname

#  --debug_skip_vtm \

recon_fname="./output/$enc_dir/recon/*"
dec_fname="./output/$0/*"

set -ex 

echo checking encoder and decoder generating matching results
#python ./tools/check_psnr.py --f1 $recon_fname --f2 $dec_fname -t 0 # Bit-exact
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $dec_fname --f2_bitdepth 10 -t 0 # Bit-exact

