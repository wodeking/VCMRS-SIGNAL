# export CUDA_VISIBLE_DEVICES=-1
# set -ex
fn_ref_md5=ref_md5_dgx1_gpu.json
outroot=./ref_data
mnt_dir=/mnt/karedgxdatastore/VCM/NN_VVC_Framework/test_ref_data

if true; then
  echo '################################################'
  echo 'Generating data for multiple images'
	export CUDA_VISIBLE_DEVICES=0
	out_dir=$outroot/images_gpu
	mkdir -p $out_dir

  fname1='../images/lighthouse.png'
  fname2='../images/baboon.jpg'
  # image encoding
  python ../../Codec/encoder.py \
				--output_dir $out_dir \
				$fname1 $fname2

  bs_fname1="$out_dir/bitstream/lighthouse.bin"
  recon_fname1="$out_dir/recon/lighthouse.png"
  bs_fname2="$out_dir/bitstream/baboon.bin"
  recon_fname2="$out_dir/recon/baboon.png"

	export CUDA_VISIBLE_DEVICES=-1
  dec_dir="$out_dir/dec_cpu"
  # image decoding
  python ../../Codec/decoder.py \
    --output_dir $dec_dir \
    $bs_fname1 $bs_fname2

  dec_fname1=$dec_dir/lighthouse.png
  dec_fname2=$dec_dir/baboon.png

  echo checking for bit-exactness of file 1
  python ./check_psnr.py --f1 $dec_fname1 --f2 $recon_fname1 -t 0

  echo echo checking for bit-exactness of file 2
  python ./check_psnr.py --f1 $dec_fname2 --f2 $recon_fname2 -t 0
  
  echo 
  echo completed
  echo
fi

if true; then
  # input video directory
  echo '################################################'
  echo 'Generating data for video coding using ini file as input'
	export CUDA_VISIBLE_DEVICES=0
	output_dir=$outroot/videos_gpu
  # encoding
  fname='ref_test.ini'
  # -m debugpy --listen 0.0.0.0:5678  
  python ../../Codec/encoder.py \
    --num_workers 3 \
		--output_dir $output_dir \
    $fname

	export CUDA_VISIBLE_DEVICES=-1
  # decoding
  bs_fname=$output_dir/bitstream/testv1_256x128.bin
  dec_dir=$output_dir/dec_cpu
  # -m debugpy --listen 0.0.0.0:5678  
  python ../../Codec/decoder.py \
    --output_dir $dec_dir \
    $bs_fname


  rec_dir=$output_dir/recon
  echo "checking for bit-exactness"
  python ./check_psnr.py --f1 $rec_dir --f2 $dec_dir -t 0
  
  echo 
  echo completed
  echo
fi

echo "generating md5 references for bit-exactness cross-device cross-checks"
python gen_md5_ref.py $outroot $fn_ref_md5

echo "Copying the generated data to $mnt_dir"
mkdir -p $mnt_dir
cp -r realpath $outroot/. $mnt_dir