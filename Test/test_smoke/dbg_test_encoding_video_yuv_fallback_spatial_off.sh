#!/bin/bash

# test encoding an video in fallback mode

set -e
fname='./videos/testv1_256x128_420p.yuv'
python -m vcmrs.encoder \
  --SourceWidth 256  \
  --SourceHeight 128 \
  --InputBitDepth 8 \
  --InputChromaFormat '420' \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --quality 57 \
  --working_dir ./output/working_dir/$0 \
  --SpatialResample "Bypass" \
  $fname

recon_fname=./output/$0/recon/$(basename $fname)
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $fname --f2_bitdepth 8 -t 8

