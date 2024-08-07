#!/bin/bash

# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

script_dir=`dirname $0`
cd $script_dir

pip install -r Requirements.txt
# install development version
pip install --no-deps -e .
