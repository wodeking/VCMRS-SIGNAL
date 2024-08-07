#!/bin/bash
# test encoding and decoding videos with various input arguments

set -e

#export CUDA_VISIBLE_DEVICES=-1
pushd `dirname $0`
test_name=$(basename $0)
output_dir="./output/${test_name%.*}"
mkdir -p $output_dir
output_dir=$(realpath $output_dir)
bs_dir='bitstream'
recon_dir='recon'
dec_dir='dec'

if true; then
  # input video directory
  echo '################################################'
  echo 'Test video coding in a directory with various input arguments'

  # encoding
  input_video='./videos/testv1_256x128'
  python -m vcmrs.encoder \
    --directory_as_video \
    --output_dir $output_dir \
    --quality 42 \
    --quality 22 \
    $input_video


  # decoding
  bs_fname=$output_dir/$bs_dir/testv1_256x128.bin
  dec_dir=$output_dir/$dec_dir
  python -m vcmrs.decoder \
    --output_dir $dec_dir \
    --output_video_format PNG \
    $bs_fname
  dec_dir=$dec_dir/testv1_256x128
  rec_dir=$output_dir/$recon_dir/testv1_256x128
  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $rec_dir --f2 $dec_dir -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $rec_dir --f2 $input_video -t 1
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Test multiple videos coding'
  prev_outdir=$output_dir
  output_dir=$output_dir/multi_test
  fname1=$(realpath './videos/testv1_256x128')
  fname2=$(realpath './videos/testv1_640x512')
  # video encoding
  python -m vcmrs.encoder \
          --directory_as_video \
          --output_dir $output_dir \
          --quality 47 \
          $fname1 $fname2

  bs_fname1=$output_dir/$bs_dir/testv1_256x128.bin
  recon_fname1=$output_dir/$recon_dir/testv1_256x128
  bs_fname2=$output_dir/$bs_dir/testv1_640x512.bin
  recon_fname2=$output_dir/$recon_dir/testv1_640x512

  python -m vcmrs.decoder \
    --output_video_format PNG \
    --output_dir $output_dir/$dec_dir $bs_fname1 $bs_fname2

  dec_fname1=$output_dir/$dec_dir/testv1_256x128
  dec_fname2=$output_dir/$dec_dir/testv1_640x512

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 1

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 1

  echo 
  echo completed
  echo
  output_dir=$prev_outdir
fi


if true; then
  echo '################################################'
  echo 'Test multiple videos coding from a .ini file'
  prev_outdir=$output_dir
  output_dir=./output/testv1_52
  fname=test3.ini
  # video encoding
  python -m vcmrs.encoder \
          $fname

  bs_fname1=$output_dir/$bs_dir/testv1_256x128.bin
  recon_fname1=$output_dir/$recon_dir/testv1_256x128
  bs_fname2=$output_dir/$bs_dir/testv1_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/testv1_1920x1080

  python -m vcmrs.decoder \
    --output_video_format PNG \
    --output_dir $output_dir/$dec_dir $bs_fname1 $bs_fname2

  fname1=$(realpath './videos/testv1_256x128')
  fname2=$(realpath './videos/testv1_1920x1080')
  dec_fname1=$output_dir/$dec_dir/testv1_256x128
  dec_fname2=$output_dir/$dec_dir/testv1_1920x1080

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 1

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 1

  echo 
  echo completed
  echo
  output_dir=$prev_outdir
fi
exit
set -e

#export CUDA_VISIBLE_DEVICES=-1
pushd `dirname $0`
test_name=$(basename $0)
output_dir="./output/${test_name%.*}_temporal_off"
mkdir -p $output_dir
output_dir=$(realpath ${output_dir})
bs_dir='bitstream'
recon_dir='recon'
dec_dir='dec'

if true; then
  # input video directory
  echo '################################################'
  echo 'Temporal off - Test video coding in a directory with various input arguments'

  # encoding
  input_video='./videos/testv1_256x128'
  python -m vcmrs.encoder \
    --directory_as_video \
    --output_dir $output_dir \
    --quality 42 \
    --quality 22 \
    --TemporalResample "Bypass" \
    $input_video


  # decoding
  bs_fname=$output_dir/$bs_dir/testv1_256x128.bin
  dec_dir=$output_dir/$dec_dir
  python -m vcmrs.decoder \
    --output_dir $dec_dir \
    --output_video_format PNG \
    --TemporalResample "Bypass" \
    $bs_fname
  dec_dir=$dec_dir/testv1_256x128
  rec_dir=$output_dir/$recon_dir/testv1_256x128
  echo checking encoder and decoder generating matching results
  python ./tools/check_psnr.py --f1 $rec_dir --f2 $dec_dir -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $rec_dir --f2 $input_video -t 1 
  
  echo 
  echo completed
  echo
fi

if true; then
  echo '################################################'
  echo 'Temporal off - Test multiple videos coding'
  prev_outdir=$output_dir
  output_dir=$output_dir/multi_test
  fname1=$(realpath './videos/testv1_256x128')
  fname2=$(realpath './videos/testv1_640x512')
  # video encoding
  python -m vcmrs.encoder \
          --directory_as_video \
          --output_dir $output_dir \
          --quality 47 \
          --TemporalResample "Bypass" \
          $fname1 $fname2

  bs_fname1=$output_dir/$bs_dir/testv1_256x128.bin
  recon_fname1=$output_dir/$recon_dir/testv1_256x128
  bs_fname2=$output_dir/$bs_dir/testv1_640x512.bin
  recon_fname2=$output_dir/$recon_dir/testv1_640x512

  python -m vcmrs.decoder \
    --output_video_format PNG \
    --output_dir $output_dir/$dec_dir \
    --TemporalResample "Bypass" \
    $bs_fname1 $bs_fname2

  dec_fname1=$output_dir/$dec_dir/testv1_256x128
  dec_fname2=$output_dir/$dec_dir/testv1_640x512

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 1

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 1

  echo 
  echo completed
  echo
  output_dir=$prev_outdir
fi


if true; then
  echo '################################################'
  echo 'Temporal off - Test multiple videos coding from a .ini file'
  prev_outdir=$output_dir
  output_dir=./output/testv1_52
  fname=test3.ini
  # video encoding
  python -m vcmrs.encoder \
         --TemporalResample "Bypass" \
          $fname

  bs_fname1=$output_dir/$bs_dir/testv1_256x128.bin
  recon_fname1=$output_dir/$recon_dir/testv1_256x128
  bs_fname2=$output_dir/$bs_dir/testv1_1920x1080.bin
  recon_fname2=$output_dir/$recon_dir/testv1_1920x1080

  python -m vcmrs.decoder \
    --output_video_format PNG \
    --output_dir $output_dir/$dec_dir \
    --TemporalResample "Bypass" \
    $bs_fname1 $bs_fname2

  fname1=$(realpath './videos/testv1_256x128')
  fname2=$(realpath './videos/testv1_1920x1080')
  dec_fname1=$output_dir/$dec_dir/testv1_256x128
  dec_fname2=$output_dir/$dec_dir/testv1_1920x1080

  echo checking encoder and decoder generating matching results for file 1
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $dec_fname1 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname1 --f2 $fname1 -t 1

  echo checking encoder and decoder generating matching results for file 2
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $dec_fname2 -t 0

  echo checking the reconstructed images has a good quality
  python ./tools/check_psnr.py --f1 $recon_fname2 --f2 $fname2 -t 1

  echo 
  echo completed
  echo
  output_dir=$prev_outdir
fi

popd
