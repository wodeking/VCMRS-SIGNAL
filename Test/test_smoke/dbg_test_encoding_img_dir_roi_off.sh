#!/bin/bash

#test encoding images in a directory

set -e

fname='./images'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --debug \
  --working_dir ./output/working_dir/$0 \
  --ROI "Bypass" \
  $fname

recon_fname=./output/$0/recon
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 8

