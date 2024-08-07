#!/bin/bash

# test encoding with ini file input

set -e
fname='test1.ini'
python -m vcmrs.encoder \
    --TemporalResample "Bypass" \
    --ROI "Bypass" \
  --SpatialResample "Bypass" \
  $fname
  
recon_fname=./output/test1.ini/recon/testv1_256x128
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 ./videos/testv1_256x128 --f2_bitdepth 8 -t 8