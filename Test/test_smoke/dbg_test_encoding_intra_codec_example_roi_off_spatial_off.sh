#!/bin/bash

# test ExampleIntraCodec component

set -e
fname='./images/lighthouse.png'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --debug \
  --quality 47 \
  --IntraCodec ExampleIntraCodec \
  --working_dir ./output/working_dir/$0 \
  --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname

recon_fname=./output/$0/recon/lighthouse.png
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 8

