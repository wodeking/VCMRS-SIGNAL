#!/bin/bash

# test encoding a video using VTM as inner codec

set -e
fname='./videos/testv1_256x128'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --InnerCodec VTM \
  --SpatialDescriptorMode UsingDescriptor \
  --SpatialDescriptor ./spatial_testv1_256x128.csv \
  --quality 42 \
  --working_dir ./output/working_dir/$0 \
  $fname

#  --debug_skip_vtm \

recon_fname=./output/$0/recon/testv1_256x128
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 1

