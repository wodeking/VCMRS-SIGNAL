#!/bin/bash

#
# encode CTC datasets
# The fast mode parallelize the task better, but cannot be used 
# to measure encoding time
#

eval "$(conda shell.bash hook)"
conda activate vcm


set -ex

# TVD dataset
echo
echo '#################################################'
echo
python t_encode_tvd_tracking.py --enc_mode LD #--test_id $test_id

echo
echo '#################################################'
echo
python t_encode_tvd_tracking.py --enc_mode RA #--test_id $test_id

echo
echo '#################################################'
echo
python t_encode_tvd_tracking.py --enc_mode AI #--test_id $test_id

# SFU dataset
echo
echo
echo '#################################################'
echo
python t_encode_sfu.py --enc_mode LD #--test_id $test_id

echo
echo '#################################################'
echo
python t_encode_sfu.py --enc_mode RA #--test_id $test_id

echo
echo '#################################################'
echo
python t_encode_sfu.py --enc_mode AI #--test_id $test_id

# image datasets
echo
echo '#######################################################'
echo
python encode_openimages.py

echo
echo '#######################################################'
echo
python encode_flir.py

echo
echo '#######################################################'
echo
python encode_tvd_detseg.py

echo $0 completed!


