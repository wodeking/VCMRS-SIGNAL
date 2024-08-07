#!/bin/bash

# test encoding a large image with SimpleDownsample spatial resampler

set -e
fname='./images/test_1920x1080.png'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --debug \
  --SpatialResample SimpleDownsample \
  --working_dir ./output/working_dir/$0 \
  $fname

recon_fname=./output/$0/recon/test_1920x1080.png
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 10

