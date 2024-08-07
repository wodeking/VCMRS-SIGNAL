# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import numpy as np
import torch
from io import BytesIO
import time


# convert data to bitstream using npz
def get_npz_stream(data):
  f = BytesIO()
  np.savez_compressed(f, x=data)
  strm = f.getvalue()

def get_stream_npz(bs):
  try:
    f = BytesIO(bs)
    data = np.load(f)
    return data['x']
  except:
    return None

# get bit stream from npz or uncompressed which every is smaller 
# data is in type of uint8, other levels is not supported
def get_bs(data):
  x = data.detach().cpu().numpy().astype(np.uint8)
  npz_bs = get_npz_stream(x) 
  pl_bs = x.reshape(-1).tobytes()
  if len(npz_bs)<len(pl_bs):
    return npz_bs
  return pl_bs  

# get data from bitstream
def get_data(bs, xs):
  data = get_stream_npz(bs)
  if data is None:
    #print('npz is not smaller. use uncompressed')
    data = np.frombuffer(bs, dtype=np.uint8)
  return data.reshape(xs)

# read a value from a file
def read_value_from_f(f, dtype=np.uint8, num=1):
  buf = f.read(np.dtype(dtype).itemsize*num)
  data = np.frombuffer(buf, dtype=dtype)
  if num==1: data=data[0]
  return data
  

