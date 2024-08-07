#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate vcm


# execute CI test
bash run_CI_tests.sh

# run smoke test cases
export CUDA_VISIBLE_DEVICES=0
for t1 in test_smoke/dbg_test_*; do
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"
done

# run smoke test cases
export CUDA_VISIBLE_DEVICES=-1
for t1 in test_smoke/dbg_test2_*; do
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"
done



# run numerical stability test
export CUDA_VISIBLE_DEVICES=0
for t1 in dbg_test3_*; do
  echo --------------------------
  echo Executing $t1
  echo 

  bash $t1

  echo  -e "\n\n\n"
done

echo
echo  "all test completed"
