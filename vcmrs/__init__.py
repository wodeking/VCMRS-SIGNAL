# This file is covered by the license agreement found in the file “license.txt” in the root of this project.


import logging
import os
from vcmrs.Utils import colorful_logger

def setup_logger(name="vcmrs", logfile='', debug=False):
   glob_vars = globals()
   if logfile is None or logfile == '':
      logfile = None
   else:
      logfile = os.path.abspath(logfile)
      os.makedirs(os.path.dirname(logfile), exist_ok=True)
   logger = colorful_logger.setup_logger(name=name, output=logfile) # leave output as None to not save logs.
   if debug:
      logger.setLevel(logging.DEBUG) # Enable log at DEBUG level (DEBUG < INFO < WARNING < ERROR < CRITICAL = FATAL)
      for handler in logger.handlers:
         handler.setLevel(logging.DEBUG)
         
   glob_vars["logger"] = logger
   glob_vars["log"] = logger.info
   glob_vars["debug"] = logger.debug
   glob_vars["warning"] = logger.warning
   glob_vars["error"] = logger.error
   glob_vars["critical"] = logger.critical

   return logger
