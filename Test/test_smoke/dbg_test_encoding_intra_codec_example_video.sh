#!/bin/bash

# test encoding a video using ExampleIntraCodec component

set -e
fname='./videos/testv1_256x128'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --quality 42 \
  --IntraCodec 'ExampleIntraCodec' \
  --working_dir ./output/working_dir/$0 \
  $fname

recon_fname=./output/$0/recon/testv1_256x128
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 1

