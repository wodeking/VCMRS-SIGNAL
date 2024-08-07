#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate vcm


# execute CI test
export CUDA_VISIBLE_DEVICES=0
for t1 in ./test_CI/dbg_test_*; do
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"
done

export CUDA_VISIBLE_DEVICES=-1
for t1 in ./test_CI/dbg_test2_*; do
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"
done

# run numerical stability test
export CUDA_VISIBLE_DEVICES=0
t1=dbg_test3_crossplatform_deterministic.sh
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"

# execute CI specific test cases
export CUDA_VISIBLE_DEVICES=0
for fname in test_CI*.sh; do
  echo
  echo running $fname
  echo
  # clean output directory
  rm -rf ./output/*
  bash $fname
done

echo
echo  "all test completed"
