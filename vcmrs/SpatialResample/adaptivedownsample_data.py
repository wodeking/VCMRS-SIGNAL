# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

scale_factor_id_mapping = {
      1.0 : 0,
      0.90 : 1,
      0.70 : 2,
      0.50 : 3,
      0.30 : 4
}

id_scale_factor_mapping = dict(reversed(x) for x in scale_factor_id_mapping.items())

corr_thresh = 0.9
top_rank_rate=0.7