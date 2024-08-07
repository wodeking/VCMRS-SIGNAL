#!/bin/bash

# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# install vcm decoder package

set -e

# recommended environment
echo ' 
  Recommnended environment: 
    CUDA: 11.3
    python 3.8.13
    torch 1.12.0+cu113
    torchvision 0.13.0+cu113
'

# Download pretrained models
if [ -z "$MPEG_CONTENT_PASSWORD" ]; then
  read -sp "Please enter MPEG document system password: " MPEG_CONTENT_PASSWORD
  echo
  echo "Checking MPEG password (please wait)..."
  set +e
  wget -q --no-check-certificate --spider --tries=1 --user mpeg --password="$MPEG_CONTENT_PASSWORD" https://content.mpeg.expert/data/
  if [ $? -ne 0 ]; then
    echo 
    echo
    echo "The entered MPEG document system password is not correct. No pretrained model will be downloaded"
    echo
    unset MPEG_CONTENT_PASSWORD
  fi
  set -e
fi

if [ -n "$MPEG_CONTENT_PASSWORD" ]; then
  MODEL_URL=https://content.mpeg.expert/data/MPEG-AI/VCM/VCMRS/v0.1/Pretrained
  MODEL_DST_DIR=./vcmrs/InnerCodec/NNManager/Pretrained
  wget -q -N --no-check-certificate -c --user mpeg --password "$MPEG_CONTENT_PASSWORD" $MODEL_URL/intra_human_adapter_A1.pth.tar -P $MODEL_DST_DIR
  wget -q -N --no-check-certificate -c --user mpeg --password "$MPEG_CONTENT_PASSWORD" $MODEL_URL/inter_machine_adapter_A3_wo_inject_w01.pth.tar -P $MODEL_DST_DIR
  wget -q -N --no-check-certificate -c --user mpeg --password "$MPEG_CONTENT_PASSWORD" $MODEL_URL/inter_machine_adapter_A0_w001.pth.tar -P $MODEL_DST_DIR
  wget -q -N --no-check-certificate -c -np -nH --cut-dirs=100 -r -R "index.html*" --user mpeg --password "$MPEG_CONTENT_PASSWORD" $MODEL_URL/intra_codec/ -P $MODEL_DST_DIR/intra_codec
  MODEL_DST_DIR=./vcmrs/TemporalResample/models
  wget -q -N --no-check-certificate -c --user mpeg --password "$MPEG_CONTENT_PASSWORD" $MODEL_URL/flownet.pkl -P $MODEL_DST_DIR
fi

# check md5sum of downloaded models
md5sum --strict --check pretrained.chk
if [ $? -ne 0 ]; then
  echo "The pretrained models are not downloaded correctly. This problem may be fixed by running this script again"
  exit 1
fi


env_name=vcm
# Run setup_vcm_env.sh to setup VCM environment for conda. 
bash setup_vcm_env.sh $env_name

# activate vcm_decoder environment
eval "$(conda shell.bash hook)"
conda activate $env_name

set -x

# install RoI related packages

# download JDE model
pip install gdown


JDE_MODEL_FILE='jde.1088x608.uncertainty.pt'
JDE_MODEL_MD5='d73a0ae671c97a84afb14cfa7994504a'
JDE_MODEL_PATH='./vcmrs/ROI/roi_generator'

if [ -f $JDE_MODEL_FILE ] && [ $(md5sum $JDE_MODEL_FILE | cut -f 1 -d ' ') = $JDE_MODEL_MD5 ]; then
  echo "found JDE model in root folder, moving to:" $JDE_MODEL_PATH
  mv ./$JDE_MODEL_FILE  $JDE_MODEL_PATH/$JDE_MODEL_FILE
fi

#primary link
pushd $JDE_MODEL_PATH
if [ ! -f $JDE_MODEL_FILE ] || [ ! $(md5sum $JDE_MODEL_FILE | cut -f 1 -d ' ') = $JDE_MODEL_MD5 ]; then
  echo "downloading JDE model from primary link...."
  set +e
  gdown -c 1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA  # links to https://drive.google.com/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA
  set -e
fi

#secondary link
if [ ! -f $JDE_MODEL_FILE ] || [ ! $(md5sum $JDE_MODEL_FILE | cut -f 1 -d ' ') = $JDE_MODEL_MD5 ]; then
  echo "downloading JDE model from secondary link...."
  set +e
  gdown -c 1Wj5m_gOuMcT7YErIg9oI66FNJUKwC97r  # links to https://drive.google.com/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA
  set -e
fi

#tertiary link
if [ ! -f $JDE_MODEL_FILE ] || [ ! $(md5sum $JDE_MODEL_FILE | cut -f 1 -d ' ') = $JDE_MODEL_MD5 ]; then
  echo "downloading JDE model from tertiary link...."
  set +e  
  wget -q -O "$JDE_MODEL_FILE" "https://www.dropbox.com/scl/fi/8bd0b6kuwv4k0dyi17tsn/jde.1088x608.uncertainty.pt?rlkey=yl6gggnn5o6up07l3nvwpy7ao&dl=0"
  set -e
fi

if [ ! -f $JDE_MODEL_FILE ] || [ ! $(md5sum $JDE_MODEL_FILE | cut -f 1 -d ' ') = $JDE_MODEL_MD5 ]; then
  echo "Unable to find nor download jde.1088x608.uncertainty.pt model"
  echo "Please download it manually from:"
  echo "https://drive.google.com/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA"
  echo "and place inside path: " $JDE_MODEL_PATH
  popd
  exit -1  
fi
popd


python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
# downgrad pillow
pip install pillow==9.5.0


# download pretrained model
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl -P ./vcmrs/ROI/roi_generator/weights/


pip install Cython==3.0.0

# download pretrained model (yolov7)
pushd ./vcmrs/SpatialResample/models
mkdir -p weights
rm -rf yolov7 weights/*
wget --no-check-certificate https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P ./weights
git clone https://github.com/WongKinYiu/yolov7.git
rm -rf yolov7/requirements.txt
popd

# install python packages
pip install -r Requirements.txt

# install VCMRS
pip install --no-deps -e .

# compile VTM
pushd ./vcmrs/InnerCodec/VTM
mkdir -p build
cd build
cmake ..
make -j 8
popd

# Install the LIC intra codec package
pushd ./vcmrs/InnerCodec/NNManager/NNIntraCodec/LIC/e2evc
bash install.sh
popd

# install testing image and video
pushd Test
python tools/gen_video.py # generate test video with resolution 256x128
python tools/gen_video.py --width 1920 --height 1080 --length 65
python tools/gen_video.py --width 640 --height 512 --length 5
python tools/gen_video.py --width 257 --height 128 --length 5
python tools/gen_image.py --width 637 --height 517 
python tools/gen_image.py --width 638 --height 517 
python tools/gen_image.py --width 1920 --height 1080 
#python tools/gen_image.py --width 2560 --height 1600
popd
