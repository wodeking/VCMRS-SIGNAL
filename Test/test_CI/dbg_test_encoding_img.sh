#!/bin/bash

# test encoding an image

set -e
bname='lighthouse'
bname='test_1920x1080'
fname="./images/$bname.png"
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --debug \
  --working_dir ./output/working_dir/$0 \
  $fname

recon_fname=./output/$0/recon/$bname.png
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 10

