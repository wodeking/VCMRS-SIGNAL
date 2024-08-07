#!/bin/bash

# test encoding and decoding on different HW platformats, e.g., CPU and GPU

set -e

function check_md5sum {
    file1="$1"
    file2="$2"

    md5sum_file1=$(md5sum "$file1" | awk '{print $1}')
    md5sum_file2=$(md5sum "$file2" | awk '{print $1}')

    # Compare the checksums
    if [ "$md5sum_file1" = "$md5sum_file2" ]; then
        echo 0
    else
        echo 1
    fi
}

fname='../Test/videos/testv1_256x128_420p.yuv'
yuv_md5=()
bin_md5=()

QP_VALUES=(42 50)
ENC_DEVICE_IDS=(gpu cpu)
DEC_DEVICE_IDS=(gpu cpu)
CONFIGS=("AllIntra" "RandomAccess" "LowDelay")

# Loop over all possible combinations of QP, device and config
for QP in "${QP_VALUES[@]}"; do
  for enc_device in "${ENC_DEVICE_IDS[@]}"; do
    for dec_device in "${DEC_DEVICE_IDS[@]}"; do
      for cvc_config in "${CONFIGS[@]}"; do
        if [ $enc_device = gpu ]; then
          enc_cuda=0
        else
          enc_cuda=-1
        fi
        if [ $dec_device = gpu ]; then
          dec_cuda=0
        else
          dec_cuda=-1
        fi

        CUDA_VISIBLE_DEVICES=$enc_cuda python -m vcmrs.encoder \
          --SourceWidth 256  \
          --SourceHeight 128 \
          --InputBitDepth 8 \
          --InputChromaFormat '420' \
          --output_dir ./output/$0 \
          --directory_as_video \
          --quality $QP \
          --Configuration $cvc_config \
          --FramesToBeEncoded 5 \
          --FrameSkip 0 \
          --working_dir ./output/working_dir/$0 \
          --output_bitstream_fname "bitstream/{bname}_qp${QP}_${cvc_config}_e${enc_device}.bin" \
          --output_recon_fname "recon/{bname}_qp${QP}_${cvc_config}_e${enc_device}" \
          $fname

        encoded_bin=output/$0/bitstream/testv1_256x128_420p_qp${QP}_${cvc_config}_e${enc_device}.bin

        decoded_yuv=testv1_256x128_420p_qp${QP}_${cvc_config}_e${enc_device}_d${dec_device}

        echo start decoding ...
        CUDA_VISIBLE_DEVICES=$dec_cuda python -m vcmrs.decoder \
        --working_dir output/temp/$0/testv1_256x128_420p \
        --debug \
        --output_dir ./output/decode/$0/recon \
        --output_recon_fname $decoded_yuv \
        $encoded_bin

        decoded_yuv=./output/decode/$0/recon/${decoded_yuv}.yuv
        echo  Decoding completed

        

        # Check YUV exactness between encoder and decoder
        chk_rslt=$(check_md5sum ./output/$0/recon/testv1_256x128_420p_qp${QP}_${cvc_config}_e${enc_device}.yuv $decoded_yuv)

        echo "Md5 check: ${decoded_yuv}"
        if [ $chk_rslt -eq 0 ]; then
            echo "Reconstructed YUV file (encoder) matches decoded YUV file"
            yuv_md5+=($(md5sum "$decoded_yuv" | awk '{print $1}'))
            bin_md5+=($(md5sum "$encoded_bin" | awk '{print $1}'))
        else
            echo "Reconstructed YUV file (encoder) DOES NOT match decoded YUV file"
            exit 1
        fi

      done
    done
  done
done

# echo "Bitstream md5s:"
# echo "${bin_md5[@]}"

# echo "YUV md5s:"
# echo "${yuv_md5[@]}"
echo "-----------------------"
echo
echo "Verifying md5 checksums of bitstreams and YUV files of the same coding configurations..."
echo
python dbg_verify_crossplatform_checksum.py "`echo ${bin_md5[@]}`" "`echo ${yuv_md5[@]}`" "`echo ${QP_VALUES[@]}`" "`echo ${CONFIGS[@]}`" "`echo ${ENC_DEVICE_IDS[@]}`" "`echo ${DEC_DEVICE_IDS[@]}`" 

