import glob
import os
from pathlib import Path
import sys
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import tvd_tracking_config, sfu_config

log_dir=sys.argv[1] #./output/decode/SFU_RA/decoding_log
time_type=sys.argv[2] # "encoding_details" or "decoding_details"

image_dataset=False
logfiles = glob.glob(os.path.join(log_dir,"*.log"))

if image_dataset:
   times = utils.collect_coding_times(logfiles, time_type=time_type, file_regx="(.*?)(qp\d+).*")
else:
   if time_type == "encoding_details":
      file_regex = "encoding_(.*?)_(qp\d+).*"
   else:
      file_regex = "(.*?)_(qp\d+).*"
   seq_order = list(tvd_tracking_config.fr_dict.keys()) + list(sfu_config.fr_dict.keys())  + [v[1].rsplit("_", 1)[0] for v in sfu_config.seq_dict.values()]
   times = utils.collect_coding_times(logfiles, time_type=time_type, file_regx=file_regex, seq_order_list=seq_order, in_hours=False)
grouped = times.groupby(["sequence","qp"], sort=False)
if time_type == "encoding_details":
   count = grouped.count()["encoding_VTM_time"]
else:
   count = grouped.count()["decoding_VTM_time"]
sum_times = grouped.sum(numeric_only=True)
sum_times["n_records"] = count
out_fp= os.path.join(log_dir, f"00sum_{time_type}time.csv")
sum_times.to_csv(out_fp)

print(sum_times)
print(f"Collected to {str(Path(out_fp).absolute())}.")
# %%
