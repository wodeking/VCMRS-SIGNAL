# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
from .CodeIFNet import *
from e2evc.Utils import ctx

device = torch.device("cpu")
    
class Model:
    def __init__(self):
        self.flownet = IFNet()
        self.device()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)
        if torch.cuda.is_available():
            self.flownet = self.flownet.to("cuda")

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v.double()
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location='cpu')))

    @ctx.int_conv()
    def inference(self, img0, img1, scale=1.0):
        if torch.cuda.is_available():
            img0 = img0.to('cuda')
            img1 = img1.to('cuda')
        else:
            img0 = img0.to('cpu')
            img1 = img1.to('cpu')
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        with ctx.int_conv():
            flow, mask, merged = self.flownet(imgs, scale_list)
        return merged[2]
