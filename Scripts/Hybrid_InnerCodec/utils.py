# process open images

import os
import glob
import re
import numpy as np
import cv2
import hashlib


import subprocess

from multiprocessing import Pool, Manager, Array
from functools import partial

import pandas as pd

def start_process(process_info, wait=False, env=None):
  '''
  Start a process

  Args:
    process_info: the name and arguments of the executeble, in a list
    wait: wait until the process is completed

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  if wait:
    proc = subprocess.Popen(map(str, process_info),
      stderr=subprocess.STDOUT,
      stdout=subprocess.PIPE, 
      env=env) 
    while True:
      line = proc.stdout.readline()
      print(line.decode('utf-8'), end='')
      if not line: break
    #proc.wait()
    outs, errs = proc.communicate() # no timeout is set
    if proc.returncode != 0:
      print(outs.decode('utf-8'))
    return proc.returncode 
  else:   
    proc = subprocess.Popen(map(str, process_info), env=env)
    return proc

def foo(dev_dict, lock, process_info): #usage, info):
    # select a device with the minimal usage
    usage_min = 1E5 # large enough number
    avail_dev = -1
    lock.acquire()
    for dev, usage in dev_dict.items():
      if usage < usage_min:
        avail_dev = dev
        usage_min = usage 
    dev_dict[avail_dev] += 1
    lock.release()

    print(f'processing on dev {avail_dev} usage {usage_min} ', process_info)
    proc_env = os.environ.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = str(avail_dev)
    err = start_process(process_info, wait=True, env=proc_env)
    lock.acquire()
    dev_dict[avail_dev] -= 1
    lock.release()
    return err

class GPUPool:
  def __init__(self, device_ids, proc_per_dev=1):
    self.proc_per_dev = proc_per_dev
    self.manager = Manager()
    self.dev_dict = self.manager.dict()
    self.lock = self.manager.Lock()
    for dev in device_ids:
      self.dev_dict[dev] = 0

  def process_tasks(self, tasks):
    p = Pool(len(self.dev_dict) * self.proc_per_dev)
    print(len(self.dev_dict) * self.proc_per_dev)
    ret = p.map(partial(foo, self.dev_dict, self.lock), tasks)
    return ret

def collect_coding_times(log_files, time_type="encoding", seq_order_list=None, in_hours=True, file_regx="encoding_(.*?)_(qp\d+).*"):
  log_dir = os.path.dirname(log_files[0])
  
  #regular encoding time and decoding time collection
  if time_type == "encoding" or time_type == "decoding":
    
    if time_type == "encoding":
      time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?Encoding completed in ([\d.]+) seconds")
    elif time_type == "decoding":
      time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?Decoding completed in ([\d.]+) seconds")
    else:
      raise ValueError(f"Time type {time_type} extraction is not supported.")
    
    file_regx = re.compile(file_regx)
    deno = 3600 if in_hours else 1
    times = []
    for logfile in log_files:
      with open(logfile) as f:
        lines = f.readlines()
      match = file_regx.match(os.path.basename(logfile))
      if match is None: continue
      seq, qp = match.groups()
      for line in reversed(lines):
        if (match := time_regx.match(line)): 
          times.append({
            "sequence": seq,
            "qp": qp,
            f"{time_type}_time": float(match.group(1))/deno
          })
          break

    times = pd.DataFrame(times)
    if seq_order_list is not None:
      times['seq_order'] = times['sequence'].apply(lambda x: seq_order_list.index(x))
      times = times.sort_values(by=['seq_order', 'qp'])
      times = times.drop(columns=['seq_order'])
    else:
      times = times.sort_values(['sequence', 'qp'])
    times.to_csv(os.path.join(log_dir, "00codingtime.csv"), index=False)
    return times
  
  #detailed coding time collection from encoding pipline
  elif time_type == "encoding_details":

    #create dataframe
    file_regx = re.compile(file_regx)
    times_all = []
    for logfile in log_files:
      with open(logfile) as f:
        lines = f.readlines()
      match = file_regx.match(os.path.basename(logfile))
      if match is None: continue
      seq, qp = match.groups()
      times_all.append({
        "sequence": seq,
        "qp": qp,
      })
    times_all = pd.DataFrame(times_all)

    time_type_list = ["encoding_temporal_resample", "encoding_spatial_resample", "encoding_roi", "encoding_bit_truncation", "encoding_VTM", "decoding_roi", "decoding_spatial_resample", "decoding_temporal_resample", "decoding_postfilter", "decoding_bit_truncation"]
    
    for time_type_module in time_type_list:
      if time_type_module == "encoding_VTM":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[.*?\] Inner encoding done\. Time = ([\d.]+)\(s\)")
      elif time_type_module == "encoding_temporal_resample":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[TemporalResample at encoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "encoding_spatial_resample":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[SpatialResample at encoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "encoding_bit_truncation":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[BitDepthTruncation at encoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "encoding_roi":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[ROI at encoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "decoding_temporal_resample":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[TemporalResample at decoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "decoding_spatial_resample":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[SpatialResample at decoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "decoding_bit_truncation":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[BitDepthTruncation at decoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "decoding_roi":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[ROI at decoder done\]. Time = ([\d.]+)\(s\)")
      elif time_type_module == "decoding_postfilter":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[PostFilter at decoder done\]. Time = ([\d.]+)\(s\)")
      else:
        raise ValueError(f"Time type {time_type_module} extraction is not supported.")

      file_regx = re.compile(file_regx)
      deno = 3600 if in_hours else 1
      times = []
      for logfile in log_files:
        with open(logfile) as f:
          lines = f.readlines()
        match = file_regx.match(os.path.basename(logfile))
        if match is None: continue
        seq, qp = match.groups()
        for line in reversed(lines):
          if (match := time_regx.match(line)): 
            times.append({
              "sequence": seq,
              "qp": qp,
              f"{time_type_module}_time": float(match.group(1))/deno
            })
            break

      times = pd.DataFrame(times)
      times_all[f"{time_type_module}_time"] = times[f"{time_type_module}_time"]
       
    if seq_order_list is not None:
      times_all['seq_order'] = times_all['sequence'].apply(lambda x: seq_order_list.index(x))
      times_all = times_all.sort_values(by=['seq_order', 'qp'])
      times_all = times_all.drop(columns=['seq_order'])
    else:
      times_all = times_all.sort_values(['sequence', 'qp'])
    times_all.to_csv(os.path.join(log_dir, "00codingtime.csv"), index=False)
    return times_all
  
  #inner decoder time collection
  elif time_type == "decoding_details":

    #create dataframe
    file_regx = re.compile(file_regx)
    times_all = []
    for logfile in log_files:
      with open(logfile) as f:
        lines = f.readlines()
      match = file_regx.match(os.path.basename(logfile))
      if match is None: continue
      seq, qp = match.groups()
      times_all.append({
        "sequence": seq,
        "qp": qp,
      })
    times_all = pd.DataFrame(times_all)

    time_type_list = ["decoding_VTM"]
    
    for time_type_module in time_type_list:
      if time_type_module == "decoding_VTM":
        time_regx = re.compile("\[.*?\d+:\d+:\d+\].*?\[.*?\] Inner decoding done\. Time = ([\d.]+)\(s\)")
      else:
        raise ValueError(f"Time type {time_type_module} extraction is not supported.")

      file_regx = re.compile(file_regx)
      deno = 3600 if in_hours else 1
      times = []
      for logfile in log_files:
        with open(logfile) as f:
          lines = f.readlines()
        match = file_regx.match(os.path.basename(logfile))
        if match is None: continue
        seq, qp = match.groups()
        for line in reversed(lines):
          if (match := time_regx.match(line)): 
            times.append({
              "sequence": seq,
              "qp": qp,
              f"{time_type_module}_time": float(match.group(1))/deno
            })
            break

      times = pd.DataFrame(times)
      times_all[f"{time_type_module}_time"] = times[f"{time_type_module}_time"]
       
    if seq_order_list is not None:
      times_all['seq_order'] = times_all['sequence'].apply(lambda x: seq_order_list.index(x))
      times_all = times_all.sort_values(by=['seq_order', 'qp'])
      times_all = times_all.drop(columns=['seq_order'])
    else:
      times_all = times_all.sort_values(['sequence', 'qp'])
    times_all.to_csv(os.path.join(log_dir, "00codingtime.csv"), index=False)
    return times_all

def check_md5sum(fname, md5sum):
  cmd=['md5sum', fname]
  proc = subprocess.Popen(map(str, cmd),
      stderr=subprocess.STDOUT,
      stdout=subprocess.PIPE)
  out, err = proc.communicate()
  assert out.decode('ascii').split()[0].strip() == md5sum, \
    print("Data md5sum does not match", fname, md5sum)


def check_roi_descriptor_file(data_dir,hexdigest='d795e40930ad08c8528d0709db9f1ada'):
  # this function check if the ROI descriptor files are found in a correct location
  # return false if ROI descriptor files are not found
  roi_dir = os.path.join(data_dir, 'roi_descriptors')
  assert os.path.isdir(roi_dir), \
    f'ROI descriptor files shall be stored in roi_descriptors directory in {data_dir}'
  hash = hashlib.md5()
  for f1 in sorted(glob.glob(os.path.join(roi_dir, '*.txt'))):
    hash.update(open(f1, 'rb').read())
  #print(hash.hexdigest())
  calculated_hexdigest = hash.hexdigest()
  assert calculated_hexdigest == hexdigest, \
    f'The hash of the ROI descriptor files does not match. It might be the files are corrupted or outdated. Calculated:{calculated_hexdigest} Expected:{hexdigest}'
   
