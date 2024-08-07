#!/bin/bash

# test decoding a video in LD mode and fallback mode

set -e
enc_dir=$(echo $0 | sed s/test2_decoding/test_encoding/)
fname="./output/$enc_dir/bitstream/*"
#fname='./output/dbg_test_encoding_video_yuv_ld_fallback.sh/bitstream/testv1_640x512_420p.bin'
python -m vcmrs.decoder \
  --output_dir ./output/$0 \
  --debug \
  --working_dir ./output/working_dir/$0 \
  --TemporalResample "Bypass" \
  $fname

recon_fname="./output/$enc_dir/recon/*"
dec_fname="./output/$0/*"

echo checking encoder and decoder generating matching results
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $dec_fname --f2_bitdepth 10 -t 0 # Bit-exact


