#!/bin/bash 

# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# set up conda environment for VCM codec
conda create --name $1 python=3.8.13 -y

eval "$(conda shell.bash hook)"
conda activate $1

# install torch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113


