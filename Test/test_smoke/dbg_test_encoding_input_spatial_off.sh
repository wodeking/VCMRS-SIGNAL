#!/bin/bash

# test encoding with ini file input

set -e
fname='test1.ini'
python -m vcmrs.encoder \
  --output_dir ./output/$0 \
  --SpatialResample "Bypass" \
  $fname
