#!/bin/bash

# test decoding a video to YUV format using an example intra codec

set -e
enc_dir=$(echo $0 | sed s/test2_decoding/test_encoding/)
fname="./output/$enc_dir/bitstream/*"
recon_fname="./output/$enc_dir/recon/*"
dec_fname="./output/$0/*"

python -m vcmrs.decoder \
    --output_dir ./output/$0 \
    --working_dir "./output/working_dir/$0" \
    --IntraCodec ExampleIntraCodec \
    --debug \
  --SpatialResample "Bypass" \
    --ROI "Bypass" \
    $fname

echo checking encoder and decoder generating matching results
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $dec_fname --f2_bitdepth 10 -t 0 # Bit-exact

