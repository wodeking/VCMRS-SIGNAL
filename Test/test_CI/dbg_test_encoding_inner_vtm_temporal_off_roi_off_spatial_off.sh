#!/bin/bash

# test encode an image using VTM inner codec

set -e
fname='./images/lighthouse.png'
fname='./images/test_638x517.png'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --InnerCodec VTM \
  --quality 42 \
  --working_dir ./output/working_dir/$0 \
  --TemporalResample "Bypass" \
  --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname

#  --debug_skip_vtm \

recon_fname=./output/$0/recon/test_638x517.png
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 8

