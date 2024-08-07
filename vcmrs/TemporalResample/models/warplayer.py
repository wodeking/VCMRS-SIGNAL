# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

def warp_new(tenInput, tenFlow):

    # Fix: 
    # torch linspace is not stable on different platform. The implementation is switched to numpy. However
    # a correct fix is required here. 

    #tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device='cpu').view(
    #    1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1).to('cpu')
    #tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device='cpu').view(
    #    1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]).to('cpu')

    tenHorizontal = torch.tensor(np.linspace(-1.0, 1.0, tenFlow.shape[3])).float().view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1).to('cpu')
    tenVertical = torch.tensor(np.linspace(-1.0, 1.0, tenFlow.shape[2])).float().view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]).to('cpu')

    backwarp_tenGrid_tmp = torch.cat(
        [tenHorizontal, tenVertical], 1).to('cpu')

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :].to('cpu') / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :].to('cpu') / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid_tmp + tenFlow).permute(0, 2, 3, 1).to(device)
    # Fix: 
    # this fuction is non-deterministic on cuda. A better solution is reqiured.
    aa = torch.nn.functional.grid_sample(input=tenInput.cpu(), grid=g.cpu(), mode='nearest', padding_mode='border')
    aa = aa.to(tenInput.device)

    return aa
