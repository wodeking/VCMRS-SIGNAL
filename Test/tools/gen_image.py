#!/usr/bin/env python
# generate test video

import os
import cv2
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Generating testing video')
parser.add_argument('--width', type=int, default=3840, help='width')
parser.add_argument('--height', type=int, default=2160, help='height')
parser.add_argument('--pattern', type=str, default='none', help='image pattern')

args = parser.parse_args()

W = args.width
H = args.height


output_fname= os.path.join('images', f'test_{W}x{H}.png')
os.makedirs(os.path.dirname(output_fname), exist_ok=True)

frame = np.zeros((H,W,3), dtype=np.uint8)
cv2.putText(frame, 
    text = f'{W}x{H}',
    org = (20, 100), 
    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale = 3, 
    color = (255, 0, 0),
    thickness = 12)
cv2.imwrite(output_fname, frame)

