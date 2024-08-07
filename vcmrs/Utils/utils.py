# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# utilities

from os import path
import re
import torch
this_path = path.dirname(path.abspath(__file__))

import os
import subprocess
import psutil
from types import SimpleNamespace
import asyncio

import numpy as np
import random
import time
import vcmrs

import hashlib
import os.path

###############################################
# system utilities
def fix_random_seed(seed=0):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def make_deterministic(is_deterministic=True):
  if is_deterministic:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #CUBLAS_WORKSPACE_CONFIG=:16:8
  else:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]


  
###############################################
# system process tool

def start_process_(process_info, wait=False, log_fname=None, return_out=False):
  '''
  Start a process - internal implementation

  Args:
    process_info: the name and arguments of the executeble, in a list
    wait: wait until the process is completed
    log_fname: write output to a log file
    return_out: return outputs and error_code,  else return return error_code only

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  if wait:
    proc = subprocess.Popen(map(str, process_info), 
      stderr=subprocess.PIPE, 
      stdout=subprocess.PIPE) 
    if proc is None:
      return None

    log_file = open(log_fname, 'w') if log_fname else None

    outs = ''
    while True:
      line = proc.stdout.readline().decode('utf-8')
      outs += line
      if log_file: log_file.write(line)
      if not line: break
    if outs: vcmrs.log(outs)

    if log_file: log_file.close()
 
    _, errs = proc.communicate() # no timeout is set
    if proc.returncode != 0:
      vcmrs.log('Process failed: ')
      vcmrs.log(process_info)
      vcmrs.log(errs.decode('utf-8'))

    if return_out: return proc.returncode, outs
    return proc.returncode
  else:
    proc = subprocess.Popen(map(str, process_info))
    return proc

def start_process(process_info, wait=False, log_fname=None, return_out=False):
  '''
  Start a process

  Args:
    process_info: the name and arguments of the executeble, in a list
    wait: wait until the process is completed
    log_fname: write output to a log file
    return_out: return outputs

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  for i in range(5):
    try:
      res = start_process_(process_info, wait, log_fname, return_out)
      if res is None:
        time.sleep(2) 
        continue
      return res
    except:
      time.sleep(2)
      
  return start_process_(process_info, wait, log_fname, return_out) # final iteration without "try"

def start_process_expect_returncode_0(process_info, wait=False, log_fname=None, return_out=False):
  '''
  Start a process with expectation of retrieving returncode=0

  Args:
    process_info: the name and arguments of the executeble, in a list
    wait: wait until the process is completed
    log_fname: write output to a log file
    return_out: return outputs

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  for i in range(5):
    res = start_process(process_info, wait, log_fname, return_out)
    if return_out:
      if res[0]==0: return res
    else:
      if res==0: return res
    time.sleep(2)
  return res

def run_process_(process_info):
  '''
  Run a process and return the ouptuts - internal implementation

  Args:
    process_info: the name and arguments of the executeble, in a list

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  proc = subprocess.Popen(map(str, process_info), 
      stderr=subprocess.STDOUT, 
      stdout=subprocess.PIPE) 

  outs, errs = proc.communicate() # no timeout is set
  return proc.returncode, outs.decode('utf-8')


def run_process(process_info):
  '''
  Run a process and return the ouptuts

  Args:
    process_info: the name and arguments of the executeble, in a list

  Return:
    if wait, returns the error code. Otherwise, return the proc object
  '''
  for i in range(5):
    try:
      res = run_process_(process_info)
      return res
    except:
      time.sleep(2)
  return run_process_(process_info) # final iteration without "try"
  
  
async def start_process_async(process_info, log_fname=None, time_tag=None):
  #proc = await asyncio.create_subprocess_shell(
  vcmrs.debug(process_info)
  proc = await asyncio.create_subprocess_exec(
     *(map(str, process_info)),
     stdout=asyncio.subprocess.PIPE,
     stderr=asyncio.subprocess.PIPE)

  stdout, stderr = await proc.communicate()
  out = stdout.decode()
  err = stderr.decode()
  vcmrs.log(out)
  vcmrs.log(err)
  if time_tag:
    last_line = list(filter(None,out.split('\n')))[-1]
    match = re.match("Total Time:.*?([0-9.]+)\s*sec.\s*\[elapsed\]", last_line.strip())
    if match:
      time = float(match.groups()[0])
      vcmrs.log(f"{time_tag}. Time = {time:.6}(s)")
  if log_fname:
    with open(log_fname, 'w') as f:
      f.write(out)
      f.write(err)
  return proc.returncode
  
  

def is_process_running(process_name):
  '''
  Check if a process is already running in the system
  '''
  # check if nnmanager is already started
  for proc in psutil.process_iter():
    try:
      # Check if process name contains the given name string.
      if process_name.lower() in proc.name().lower():
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return False
 
######################################################
# misc
def getattr_by_path(obj, attr_str, default=False):
  """Find the attr with the given attribute path, e.g., `'parent.child1.child2'`"""
  attr_path = attr_str.split(".")
  for attr in attr_path:
    obj = getattr(obj, attr, False)
    if not obj:
      return default
  return obj

######################################################
# calculates md5 checksum of all Python .py files inside given project folder and prepares summarized report

def get_project_checksums(root = None, sub_path="", level=0, combined_md5 = None, combined_report = None):

  if root is None: root  = os.path.dirname(vcmrs.__file__)
  
  if combined_md5 is None: combined_md5 = hashlib.md5()
  
  if combined_report is None: combined_report = []
  
  full_path = os.path.join(root, sub_path)

  dirs = []
  files = []
  for file_name in os.listdir(full_path):
    full_file_name = os.path.join(full_path, file_name)
    if os.path.isdir(full_file_name): dirs.append(file_name)
    if os.path.isfile(full_file_name) and file_name.endswith('.py'): files.append(file_name)
    
  dirs = sorted(dirs)
  files = sorted(files)
  
  for file_name in files:
  
    full_file_name = os.path.join(full_path, file_name)
    try:
      md5 = hashlib.md5(open(full_file_name, 'rb').read())
      hash_txt = str(md5.hexdigest())
      if combined_md5 is not None:
        combined_md5.update(md5.digest())
    except:
      hash_txt = "????????????????????????????????"
    
    combined_report.append( hash_txt+"\t" + os.path.join(sub_path,file_name) )
    
  for file_name in dirs:
    get_project_checksums(root, os.path.join(sub_path,file_name), level+1, combined_md5, combined_report )
  
  if (level==0):
    combined_report.append( combined_md5.hexdigest() + "\t==Combined==")
    return combined_report
