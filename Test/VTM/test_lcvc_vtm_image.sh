#!/bin/bash

set -ex

#TVD
bname=000339
fname=../Data/tvd_object_detection_dataset/$bname.png

#OpenImages
bname=ff0a5408ed87f654
fname=../Data/OpenImages/validation/$bname.jpg

qp=47
bs_fname=./output/vtm_lcvc/bitstream/$bname.bin
recon_fname=./output/vtm_lcvc/recon/$bname.png
working_dir=./output/working_dir/vtm_lcvc

mkdir -p $working_dir
mkdir -p $(dirname $bs_fname)
mkdir -p $(dirname $recon_fname)

W=$(identify -format '%w' $fname)
H=$(identify -format '%h' $fname)

W2=$(echo "($W+1) / 2 * 2" | bc)
H2=$(echo "($H+1) / 2 * 2" | bc)

#padding
pad_fname=$working_dir/${bname}_pad.png
ffmpeg -y -i $fname -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  $pad_fname

# to YUV
yuv_fname=$working_dir/${bname}_${W2}x${H2}.yuv
ffmpeg -y -i $pad_fname -f rawvideo -pix_fmt yuv420p -dst_range 1  $yuv_fname

#encoding
cfg_fname=../vcmrs/InnerCodec/VTM/cfg/encoder_intra_vtm.cfg
yuv_recon_fname=$working_dir/${bname}_recon_${W2}x${H2}.yuv
# lcvc VTM
cvc_bin=../vcmrs/InnerCodec/VTM/bin/EncoderAppStatic
# official VTM
#cvc_bin=EncoderAppStatic
$cvc_bin -c $cfg_fname \
  -i $yuv_fname \
  -o $yuv_recon_fname \
  -b $bs_fname \
  -q $qp \
  --ConformanceWindowMode=1 \
  -wdt $W2 \
  -hgt $H2 \
  -f 1 \
  -fr 30 \
  --InternalBitDepth=10

# convert YUV to png
yuv_recon_png_fname=$working_dir/${bname}_recon_${W2}x${H2}.png
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s ${W2}x${H2} -src_range 1 -i $yuv_recon_fname -frames 1 -pix_fmt rgb24 $yuv_recon_png_fname

# original size
ffmpeg -i $yuv_recon_png_fname -vf "crop=$W:$H" $recon_fname 


