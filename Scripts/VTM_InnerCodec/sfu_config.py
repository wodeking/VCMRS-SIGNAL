## Dicts for SFU-HW sequences

seq_dict = {
    "Traffic"         : ("A", "Traffic_2560x1600_30"),          # 1
    "ParkScene"       : ("B", "ParkScene_1920x1080_24"),        # 2
    "Cactus"          : ("B", "Cactus_1920x1080_50"),           # 3
    "BasketballDrive" : ("B", "BasketballDrive_1920x1080_50"),  # 4
    "BQTerrace"       : ("B", "BQTerrace_1920x1080_60"),        # 5
    "BasketballDrill" : ("C", "BasketballDrill_832x480_50"),    # 6
    "BQMall"          : ("C", "BQMall_832x480_60"),             # 7
    "PartyScene"      : ("C", "PartyScene_832x480_50"),         # 8
    "RaceHorsesC"     : ("C", "RaceHorses_832x480_30"),         # 9
    "BasketballPass"  : ("D", "BasketballPass_416x240_50"),     # 10
    "BQSquare"        : ("D", "BQSquare_416x240_60"),           # 11
    "BlowingBubbles"  : ("D", "BlowingBubbles_416x240_50"),     # 12
    "RaceHorsesD"     : ("D", "RaceHorses_416x240_30"),         # 13
    "Kimono"          : ("O", "Kimono_1920x1080_24"),           # 14
}


res_dict = {
    "A" : (2560, 1600),
    "B" : (1920, 1080),
    "C" : ( 832,  480),
    "D" : ( 416,  240),
    "E" : (1280,  720),
    "O" : (1920, 1080),
}

fr_dict = { # (IntraPeriod, FrameRate, FramesToBeEncoded, FrameSkip)
    "Traffic"         : (32, 30, 33, 117),
    "ParkScene"       : (32, 24, 33, 207),
    "Cactus"          : (64, 50, 97, 403),
    "BasketballDrive" : (64, 50, 97, 403),
    "BQTerrace"       : (64, 60, 129, 471),
    "BasketballDrill" : (64, 50, 97, 403),
    "BQMall"          : (64, 60, 129, 471),
    "PartyScene"      : (64, 50, 97, 403),
    "RaceHorsesC"     : (32, 30, 65, 235),
    "BasketballPass"  : (64, 50, 97, 403),  
    "BQSquare"        : (64, 60, 129, 471),
    "BlowingBubbles"  : (64, 50, 97, 403),
    "RaceHorsesD"     : (32, 30, 65, 235),
    "Kimono"          : (32, 24, 33, 207),
}

def __get_qp_list(start_qp, interval, NNIntraQPOffset=-5):
  return [(x, NNIntraQPOffset) for x in range(start_qp, start_qp+interval*6, interval)]

# sequence encoding configuration
#  [(quality, NNIntraQPOffset)]
seq_cfg = {
  "Traffic"          : __get_qp_list(35, 4),

  "ParkScene"        : __get_qp_list(30, 5),
  "Cactus"           : __get_qp_list(44, 2),
  "BasketballDrive"  : __get_qp_list(38, 3),
  "BQTerrace"        : __get_qp_list(38, 3),

  "BasketballDrill"  : __get_qp_list(22, 4),
  "BQMall"           : __get_qp_list(27, 5),
  "PartyScene"       : __get_qp_list(32, 4),
  "RaceHorsesC"      : __get_qp_list(27, 4),

  "BasketballPass"   : __get_qp_list(22, 4),
  "BQSquare"         : __get_qp_list(22, 4),
  "BlowingBubbles"   : __get_qp_list(22, 4),
  "RaceHorsesD"      : __get_qp_list(27, 5),

  "Kimono"           : __get_qp_list(32, 5),
}


seq_cfg_ai = {
    "Traffic"         : __get_qp_list(32, 5, 0),
    "ParkScene"       : __get_qp_list(22, 5, 0),
    "Cactus"          : __get_qp_list(22, 5, 0),
    "BasketballDrive" : __get_qp_list(22, 5, 0),
    "BQTerrace"       : __get_qp_list(22, 5, 0),
    "BasketballDrill" : __get_qp_list(22, 5, 0),
    "BQMall"          : __get_qp_list(22, 5, 0),
    "PartyScene"      : __get_qp_list(22, 5, 0),
    "RaceHorsesC"     : __get_qp_list(22, 5, 0),
    "BasketballPass"  : __get_qp_list(22, 5, 0),
    "BQSquare"        : __get_qp_list(22, 5, 0),
    "BlowingBubbles"  : __get_qp_list(22, 5, 0),
    "RaceHorsesD"     : __get_qp_list(22, 5, 0),

    "Kimono"          : __get_qp_list(32, 5, 0),
}

seq_roi_cfg_network = "faster_rcnn_X_101_32x8d_FPN_3x"
