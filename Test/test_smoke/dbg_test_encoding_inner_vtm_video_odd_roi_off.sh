#!/bin/bash

# test encoding an video with odd size using VTM inner codec 

set -e
fname='./videos/testv1_257x128'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --InnerCodec VTM \
  --quality 42 \
  --working_dir ./output/working_dir/$0 \
  --ROI "Bypass" \
  $fname

#  --debug_skip_vtm \

recon_fname=./output/$0/recon/testv1_257x128
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 8

