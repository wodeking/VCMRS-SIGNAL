# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

from types import SimpleNamespace
import copy

#########################################
# base profile
__base_profile = SimpleNamespace()

# number of scales
__base_profile.scales = 3 

# block 2x2 is equivalent to 3 steps
__base_profile.block_size = [2,2] 

# 'default' | 'random' | 'random_block' | 'entropy_map' | 'in_entropy_map'
__base_profile.mask_type = 'default' 

# 'none' | 'mixture' | 'full' | 'split'
# 'previous' mode is not supported in mask channel model
__base_profile.channel_mode = 'mixture'

# number of seeds for mixture mode
__base_profile.mixture_seeds = 2

# number of splits for split mode
__base_profile.channel_splits = 4

# number filters, deterimines the size of CNN module
__base_profile.nr_filters = 64 

# number of mixtures in the mixture model
# Not used in PMS
__base_profile.nr_mixtures = 5 

# number of resnet blocks in the CNN module
__base_profile.nr_resblocks = 5 


##########################################
# predefined pixel processing order
# dictionary: h*1000+w : [[h0,w0], [h1,w1], ...]
__base_profile.predefined_pixel_order = { \
  # [2, 2] 3 steps
  2002: [[0,0], [0,1], [1,0]],
  # [2, 4] 6 steps
  2004: [[1,2], [1,0], [0,3], [0,1], [0,2], [0,0]],
}

############################################
# training related
# quantization training strategy: 'noise', 'ste'
__base_profile.quant_train = 'noise'

#########################################
# extra profile
# best performance
extra = copy.copy(__base_profile)
extra.scales = 4
extra.block_size = [2,4]
extra.nr_filters = 128
extra.nr_mixtures = -1
extra.nr_resblocks = 5
extra.mixture_seeds = 4
