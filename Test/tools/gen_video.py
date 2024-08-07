#!/usr/bin/env python
# generate test video

import os
import cv2
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Generating testing video')
parser.add_argument('--width', type=int, default=256, help='width')
parser.add_argument('--height', type=int, default=128, help='height')
parser.add_argument('--length', type=int, default=65, help='number of frames')

args = parser.parse_args()

video_length = args.length
W = args.width
H = args.height


video_fname=f'testv1_{W}x{H}'
output_dir = os.path.join('videos', video_fname)
os.makedirs(output_dir, exist_ok=True)

for idx in range(video_length):
  frame = np.zeros((H,W,3), dtype=np.uint8)
  cv2.putText(frame, 
    text = str(idx),
    org = (20, 100), 
    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale = 3, 
    color = (255, 0, 0),
    thickness = 12)
  o_fname = os.path.join(output_dir, f'frame_{idx:03d}.png')
  cv2.imwrite(o_fname, frame)

# convert to yuv data
i_fname = os.path.join('videos', video_fname, 'frame_%03d.png')
o_fname = os.path.join('videos', f'{video_fname}_420p.yuv')
cmd = [
     'ffmpeg',
     '-y',
     '-i',  i_fname,
     '-f', 'rawvideo',
     '-pix_fmt', 'yuv420p',
     '-dst_range', '1',
     o_fname]
print(cmd)
subprocess.run(' '.join(cmd), shell=True)

o_fname = os.path.join('videos', f'{video_fname}_420p_limited.yuv')
cmd = [
     'ffmpeg',
     '-y',
     '-i',  i_fname,
     '-f', 'rawvideo',
     '-pix_fmt', 'yuv420p',
     '-dst_range', '0',
     o_fname]
print(cmd)
subprocess.run(' '.join(cmd), shell=True)


o_fname = os.path.join('videos', f'{video_fname}_420p10le.yuv')
cmd = [
     'ffmpeg',
     '-y',
     '-i',  i_fname,
     '-f', 'rawvideo',
     '-pix_fmt', 'yuv420p10le',
     '-dst_range', '1',
     o_fname]
print(cmd)
subprocess.run(' '.join(cmd), shell=True)



