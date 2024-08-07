#!/bin/bash

# this script calculates md5 checksum of output bitstreams and reconstructions

cd output

find -L SFU_* TVD_* -name '*.bin' -exec md5sum {} \; | sort -k 2 > bitstream.chk
find -L SFU_* TVD_*  -name '*.yuv' -exec md5sum {} \; | sort -k 2 > recon.chk

echo $0 completed

