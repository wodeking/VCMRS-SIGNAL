# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
from types import SimpleNamespace

class CodecContext():
  def __init__(self, args):
    # system input arguments 
    self.input_args = args

    # system input file items
    self.input_files = []

    # context for inner codec
    self.inner_ctx = SimpleNamespace()


