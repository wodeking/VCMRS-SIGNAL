#!/bin/bash
# test different TemporalScale in YUV format

set -e
fname='./videos/testv1_1920x1080_420p10le.yuv'

python -m vcmrs.encoder \
  --SourceWidth 1920  \
  --SourceHeight 1080 \
  --InputBitDepth 10 \
  --InputChromaFormat '420' \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --quality 42 \
  --working_dir ./output/working_dir/$0 \
  --TemporalScale 2 \
  $fname
  
recon_fname=./output/$0/recon/$(basename $fname)
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $recon_fname --f1_bitdepth 10 --f2 $fname --f2_bitdepth 10 -t 1

# decoding
input_dir=./videos/testv1_1920x1080
output_dir=./output/$0
bs_fname=$output_dir/bitstream/testv1_1920x1080_420p10le.bin
dec_dir=$output_dir/dec
python -m vcmrs.decoder \
  --output_dir $dec_dir \
  --output_video_format PNG \
  $bs_fname
  
echo checking the decoding images has a good quality
python ./tools/check_psnr.py --f1 $dec_dir/$(basename $fname .yuv)  --f2 $input_dir  -t 1
