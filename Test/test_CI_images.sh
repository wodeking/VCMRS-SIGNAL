#!/bin/bash
# test basic encoding and decoding for images

set -e
pushd `dirname $0`
#export CUDA_VISIBLE_DEVICES=-1
test_name=$(basename $0)
output_dir="./output/${test_name%.*}"
mkdir -p $output_dir
output_dir=$(realpath $output_dir)
bs_dir='bitstream'
recon_dir='recon'
dec_dir='dec'

if true; then
  echo '################################################'
  echo 'Test single image'
  output_dir=$output_dir/single_img
  fname=$(realpath './images/lighthouse.png')
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir \
          $fname

  bs_fname=$output_dir/$bs_dir/lighthouse.bin
  recon_fname=$output_dir/$recon_dir/lighthouse.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir $bs_fname

  dec_fname="$output_dir/$dec_dir/lighthouse.png"

  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $recon_fname --f2 $dec_fname -t 0 # Bit-exact

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 20
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Test multiple images'
  output_dir=$output_dir/multi_img
  fname1=$(realpath './images/lighthouse.png')
  fname2=$(realpath './images/test_1920x1080.png')
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir \
          $fname1 $fname2

  bs_fname1=$output_dir/$bs_dir/lighthouse.bin
  recon_fname1=$output_dir/$recon_dir/lighthouse.png
  bs_fname2=$output_dir/$bs_dir/test_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/test_1920x1080.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir $bs_fname1 $bs_fname2

  dec_fname1=$output_dir/$dec_dir/lighthouse.png
  dec_fname2=$output_dir/$dec_dir/test_1920x1080.png

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 20

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 20

  echo 
  echo completed
  echo
fi

if true; then
  # input directory
  echo '################################################'
  echo 'Test images in a directory'
  img_dir='testv1_640x512'
  output_dir=$output_dir/img_dir
  # encoding
  input_dir=$(realpath "./videos/${img_dir}")
  python -m vcmrs.encoder \
          --output_dir $output_dir \
          $input_dir
  
  python -m vcmrs.decoder \
    --output_dir $output_dir/$dec_dir/$img_dir \
    $output_dir/$bs_dir

  rec=$output_dir/$recon_dir
  dec=$output_dir/$dec_dir/$img_dir

  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $rec --f2 $dec -t 0 # Bit-exact

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $input_dir --f2 $rec -t 20
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Test multiple images from a .ini file'
  output_dir=$output_dir/multi_ini
  fname=test2.ini
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir \
          $fname

  bs_fname1=$output_dir/$bs_dir/lighthouse.bin
  recon_fname1=$output_dir/$recon_dir/lighthouse.png
  bs_fname2=$output_dir/$bs_dir/test_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/test_1920x1080.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir $bs_fname1 $bs_fname2

  fname1=$(realpath './images/lighthouse.png')
  fname2=$(realpath './images/test_1920x1080.png')

  dec_fname1=$output_dir/$dec_dir/lighthouse.png
  dec_fname2=$output_dir/$dec_dir/test_1920x1080.png

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 20

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 20

  echo 
  echo completed
  echo
fi


set -e
pushd `dirname $0`
#export CUDA_VISIBLE_DEVICES=-1
test_name=$(basename $0)
output_dir="./output/${test_name%.*}"
mkdir -p $output_dir
output_dir=$(realpath $output_dir)
bs_dir='bitstream'
recon_dir='recon'
dec_dir='dec'

if true; then
  echo '################################################'
  echo 'Test single image with temporal off'
  output_dir=$output_dir/single_img
  fname=$(realpath './images/lighthouse.png')
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir --TemporalResample "Bypass" \
          $fname

  bs_fname=$output_dir/$bs_dir/lighthouse.bin
  recon_fname=$output_dir/$recon_dir/lighthouse.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir --TemporalResample "Bypass" $bs_fname

  dec_fname="$output_dir/$dec_dir/lighthouse.png"

  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $recon_fname --f2 $dec_fname -t 0 # Bit-exact

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname --f2 $fname -t 20
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Test multiple images with temporal off'
  output_dir=$output_dir/multi_img
  fname1=$(realpath './images/lighthouse.png')
  fname2=$(realpath './images/test_1920x1080.png')
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir --TemporalResample "Bypass" \
          $fname1 $fname2

  bs_fname1=$output_dir/$bs_dir/lighthouse.bin
  recon_fname1=$output_dir/$recon_dir/lighthouse.png
  bs_fname2=$output_dir/$bs_dir/test_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/test_1920x1080.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir --TemporalResample "Bypass" $bs_fname1 $bs_fname2

  dec_fname1=$output_dir/$dec_dir/lighthouse.png
  dec_fname2=$output_dir/$dec_dir/test_1920x1080.png

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 20

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 20

  echo 
  echo completed
  echo
fi

if true; then
  # input directory
  echo '################################################'
  echo 'Test images in a directory with temporal off'
  img_dir='testv1_640x512'
  output_dir=$output_dir/img_dir
  # encoding
  input_dir=$(realpath "./videos/${img_dir}")
  python -m vcmrs.encoder \
          --output_dir $output_dir \
          --TemporalResample "Bypass" \
          $input_dir
  
  python -m vcmrs.decoder \
    --output_dir $output_dir/$dec_dir/$img_dir \
    --TemporalResample "Bypass" \
    $output_dir/$bs_dir

  rec=$output_dir/$recon_dir
  dec=$output_dir/$dec_dir/$img_dir

  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $rec --f2 $dec -t 0 # Bit-exact

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $input_dir --f2 $rec -t 20
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Test multiple images from a .ini file with temporal off'
  output_dir=$output_dir/multi_ini
  fname=test2.ini
  # image encoding
  python -m vcmrs.encoder --output_dir $output_dir \
          --TemporalResample "Bypass" \
          $fname

  bs_fname1=$output_dir/$bs_dir/lighthouse.bin
  recon_fname1=$output_dir/$recon_dir/lighthouse.png
  bs_fname2=$output_dir/$bs_dir/test_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/test_1920x1080.png

  # image decoding
  python -m vcmrs.decoder --output_dir $output_dir/$dec_dir --TemporalResample "Bypass" $bs_fname1 $bs_fname2

  fname1=$(realpath './images/lighthouse.png')
  fname2=$(realpath './images/test_1920x1080.png')

  dec_fname1=$output_dir/$dec_dir/lighthouse.png
  dec_fname2=$output_dir/$dec_dir/test_1920x1080.png

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 20

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 20

  echo 
  echo completed
  echo
fi
popd
