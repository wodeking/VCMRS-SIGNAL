#!/bin/bash

# test encoding image in odd size

set -e
fname='./images/test_638x517.png'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --debug \
  --working_dir ./output/working_dir/$0 \
  --TemporalResample "Bypass" \
  --SpatialResample "Bypass" \
  $fname

recon_fname=./output/$0/recon/test_638x517.png
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 20

