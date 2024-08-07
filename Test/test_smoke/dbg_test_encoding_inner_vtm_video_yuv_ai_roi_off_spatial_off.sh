#!/bin/bash

# test encoding video using VTM as inner codec in AI mode

set -e
fname='./videos/testv1_640x512_420p.yuv'
python -m vcmrs.encoder \
  --SourceWidth 640 \
  --SourceHeight 512 \
  --InputBitDepth 8 \
  --InputChromaFormat '420' \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --InnerCodec VTM \
  --Configuration 'AllIntra' \
  --quality 42 \
  --working_dir ./output/working_dir/$0 \
  --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname

#  --debug_skip_vtm \

recon_fname=./output/$0/recon/$(basename $fname)
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $fname --f2_bitdepth 8 -t 8

