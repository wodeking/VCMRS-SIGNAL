
#%%
import pandas as pd
import sys

bs_md5=sys.argv[1].split(" ")
yuv_md5=sys.argv[2].split(" ")
QPS=sys.argv[3].split(" ")
CONFIGS=sys.argv[4].split(" ")
DEC_DEVICE_IDS=sys.argv[5].split(" ")
ENC_DEVICE_IDS=sys.argv[6].split(" ")

i = 0
records = []
for qp in QPS:
   for enc_dev in ENC_DEVICE_IDS:
      for dec_dev in DEC_DEVICE_IDS:
         for conf in CONFIGS:
            records.append({
               "QP": qp,
               "CONFIG": conf,
               "ENC_DEV": enc_dev,
               "DEC_DEV": dec_dev,
               "BS_md5": bs_md5[i],
               "YUV_md5": yuv_md5[i]
            })
            i+=1

df = pd.DataFrame(records).sort_values(["QP","CONFIG"])
print(df)
grouped = df.groupby(['QP','CONFIG'])[['BS_md5','YUV_md5']].nunique()
if (grouped['BS_md5'] == 1).all() and (grouped['YUV_md5'] == 1).all():
    print('**All matched.**')
else:
    print('***There are mismatches.***')
# %%
