# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import *
import numpy as np
import random
import os
from e2evc.Utils import utils_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _sigmoid_stable(x):
   x2 = x*x
   x3 = x2*x
   y = 1/(2-x+x2*0.5-x3*0.16666667)
   y2 = 1-1/(2+x+x2*0.5+x3*0.16666667)
   return torch.where(x<0, y, y2)


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        utils_nn.IntConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, kernel_size=3, stride=2, padding=1),
            conv(c // 2, c, kernel_size=3, stride=2, padding=1),
            )
        self.convblock0 = nn.Sequential(
            conv(c, c, kernel_size=3, stride=1, padding=1),
            conv(c, c, kernel_size=3, stride=1, padding=1)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c, kernel_size=3, stride=1, padding=1),
            conv(c, c, kernel_size=3, stride=1, padding=1)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c, kernel_size=3, stride=1, padding=1),
            conv(c, c, kernel_size=3, stride=1, padding=1)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c, kernel_size=3, stride=1, padding=1),
            conv(c, c, kernel_size=3, stride=1, padding=1)
        )
        self.conv1 = nn.Sequential(
            utils_nn.IntTransposedConv2d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.PReLU(c // 2),
            utils_nn.IntTransposedConv2d(c // 2, 4, kernel_size=4, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            utils_nn.IntTransposedConv2d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.PReLU(c // 2),
            utils_nn.IntTransposedConv2d(c // 2, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x, flow, scale=1):

        x = F.interpolate(x, scale_factor= 1. / scale, mode="nearest-exact", recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="nearest-exact", recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="nearest-exact", recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="nearest-exact", recompute_scale_factor=False)
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)

    def forward(self, x, scale_list=[4, 2, 1], training=False):
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:channel*2]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]

        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp_new(img0, flow[:, :2])
            warped_img1 = warp_new(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        for i in range(3):
            a = mask_list[i]
            mask_list[i] = _sigmoid_stable(a)
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])

        return flow_list, mask_list[2], merged
