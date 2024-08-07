#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.realpath('../Scripts'))
import sfu_config

#SFU
bname='RaceHorsesD'
cls,fname = sfu_config.seq_dict[bname]
W,H = sfu_config.res_dict[cls]
intra_period, frame_rate, n_frames, frameskip = sfu_config.fr_dict[bname]

frame_fnames=f'../Data/SFU/{fname}/{fname}_%03d.png'
#frame_fnames=f'../Data/SFU/{fname}/*.png'

qp=47

bs_fname=f'./output/vtm_lcvc/bitstream/{bname}_{qp}/{bname}.bin'
recon_fname=f'./output/vtm_lcvc/recon/{bname}/frame_%06d.png'
working_dir='./output/working_dir/vtm_lcvc'

os.makedirs(working_dir, exist_ok=True)
os.makedirs(os.path.dirname(bs_fname), exist_ok=True)
os.makedirs(os.path.dirname(recon_fname), exist_ok=True)


#padding
#pad_fname=$working_dir/${bname}_pad.png
#ffmpeg -y -i $fname -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  $pad_fname

# to YUV
yuv_fname=os.path.join(working_dir, f'{bname}_{W}x{H}.yuv')
cmd=['ffmpeg -y',
  #'-pattern_type glob',
  '-i', frame_fnames,
  #'-vf', f"select='between(n\,{frameskip}\,{frameskip+n_frames})'",
  '-f', 'rawvideo',
  '-pix_fmt', 'yuv420p',
  '-dst_range', '1',
  yuv_fname,
]
cmd=' '.join(map(str, cmd))
print('-------------------------')
print(cmd)
os.system(cmd)

#encoding
cfg_fname="../vcmrs/InnerCodec/VTM/cfg/encoder_randomaccess_vtm.cfg"
yuv_recon_fname=f"{working_dir}/{bname}_recon_{W}x{H}.yuv"
# LCVC VTM
cvc_bin="../vcmrs/InnerCodec/VTM/bin/EncoderAppStatic"
# official VTM
#cvc_bin="EncoderAppStatic"
cmd=[cvc_bin, '-c', cfg_fname,
  '-i', yuv_fname,
  '-o', yuv_recon_fname,
  '-b', bs_fname,
  '-q', qp,
  '--ConformanceWindowMode=1',
  '-wdt', W,
  '-hgt', H,
  '-fs', frameskip,
  '-f', n_frames,
  '-fr', frame_rate,
  '--InternalBitDepth=10',
]
print('-------------------------')
print(cmd)
os.system(' '.join(map(str, cmd)))

# convert YUV to png
#yuv_recon_png_fname=f"{working_dir}/{bname}_recon/frame_%06d.png"

os.makedirs(os.path.dirname(recon_fname), exist_ok=True)
cmd=f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {W}x{H} -src_range 1 -i {yuv_recon_fname} -frames {n_frames} -pix_fmt rgb24 {recon_fname}'
print('-------------------------')
print(cmd)
os.system(cmd)

# original size
#ffmpeg -i $yuv_recon_png_fname -vf "crop=$W:$H" $recon_fname 


