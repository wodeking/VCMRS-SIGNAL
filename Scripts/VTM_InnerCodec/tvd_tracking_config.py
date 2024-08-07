## Dicts for TVD tracking sequences

fr_dict = { # (IntraPeriod, FrameRate, FramesToBeEncoded, FrameSkip, BitDepth)
    'TVD-01_1' : (64, 50, 500, 1500, 8),  # 1 
    'TVD-01_2' : (64, 50, 500, 2000, 8),  # 2
    'TVD-01_3' : (64, 50, 500, 2500, 8),  # 3
    'TVD-02_1' : (64, 50, 636, 0, 10),    # 4
    'TVD-03_1' : (64, 50, 500, 0, 10),    # 5
    'TVD-03_2' : (64, 50, 500, 500, 10),  # 6
    'TVD-03_3' : (64, 50, 500, 1000, 10), # 7
}

# checksum of input YUV sequences
md5sum = {
  'TVD-01': '1dddac6c82e5c8e59f06d283458e2db7', 
  'TVD-02': 'aad63df298fa6401c16a36ede61e9798',
  'TVD-03': '9aa26e98ac34e7da9712c3ed4677da4b',
}


def __get_qp_list(start_qp, interval, NNIntraQPOffset=-5):
  return [x for x in range(start_qp, start_qp+interval*6, interval)]


# sequence encoding configuration
#  [qualities]
seq_cfg = {
  'TVD-01_1' : __get_qp_list(20, 3), 
  'TVD-01_2' : __get_qp_list(22, 3), 
  'TVD-01_3' : __get_qp_list(22, 4), 
  'TVD-02_1' : __get_qp_list(20, 4), 
  'TVD-03_1' : __get_qp_list(34, 4), 
  'TVD-03_2' : __get_qp_list(29, 4), 
  'TVD-03_3' : __get_qp_list(22, 5), 
}


seq_cfg_ai = {
  'TVD-01_1' : __get_qp_list(22, 5, 0), 
  'TVD-01_2' : __get_qp_list(22, 5, 0), 
  'TVD-01_3' : __get_qp_list(22, 5, 0), 
  'TVD-02_1' : __get_qp_list(22, 5, 0), 
  'TVD-03_1' : __get_qp_list(22, 5, 0), 
  'TVD-03_2' : __get_qp_list(22, 5, 0), 
  'TVD-03_3' : __get_qp_list(22, 5, 0), 
}

seq_roi_cfg_network = "yolov3_1088x608"

