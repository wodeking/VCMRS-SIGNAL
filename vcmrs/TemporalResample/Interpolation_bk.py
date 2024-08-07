# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import math,os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue
import shutil
import vcmrs
from .models.pytorch_msssim import ssim_matlab
from .models.CodeIF import Model
from pathlib import Path
import vcmrs.TemporalResample.models as temporal_model
temporal_model_dir = Path(temporal_model.__path__[0])
torch.use_deterministic_algorithms(True)

def Interpolation(imgspath, interpolateDir, rate, FramesToBeRecon):
    savepath = interpolateDir
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath)

    exp = math.log2(rate)
    cnt = (FramesToBeRecon-1) // rate
    upresampler = _UpResampler_()
    upresampler.setAttr(imgspath, int(exp), savepath, 0, cnt*rate)
    upresampler.infer()
    # padding end part with the last frame if necessary
    MOD = (FramesToBeRecon-1) % rate
    if MOD != 0:
        vcmrs.log(f'mod ={MOD}')
        lastName = sorted(os.listdir(savepath))[-1]
        last = lastName.split('.')[0].split('_')[-1]
        cnt = len(last)
        for i in range(MOD):
            padName = f'%0{len(last)}d.png' % (int(last) + i + 1)
            cnt1 = len(lastName.split('.')[0]) - cnt
            padName = lastName[0:cnt1] + padName
            curpath = os.path.join(savepath, lastName)
            copyDes = os.path.join(savepath, padName)
            shutil.copy(curpath, copyDes)

class _UpResampler_:
    def __init__(self):
        warnings.filterwarnings("ignore")
        # load model
        self.device = torch.device("cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            #cancel dynamic algorithm
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        self.model = Model()
        self.model.load_model(temporal_model_dir, -1)
        self.model.eval()
        self.model.device()
        vcmrs.log('Temporal model loaded.')
    
    def setAttr(self, img, exp, savepath, startIdx, endIdx):
        self.cnt = 0
        self.img = img
        self.exp = int(exp)
        self.savepath = savepath
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.maxLen = int((self.endIdx - self.startIdx) // (2**self.exp)) + 1

    def saveImg(self, item):
        tmpname = os.path.join(self.savepath, 'frame_{:0>6d}.png'.format(self.cnt))
        cv2.imwrite(tmpname, item[:, :, ::-1])
        self.cnt += 1

    def build_read_buffer(self, read_buffer, videogen):
        try:
            for frame in videogen:
                if not self.img is None:
                    frame = cv2.imread(os.path.join(self.img, frame))[:, :, ::-1].copy()
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(self, I0, I1, n):
        middle = self.model.inference(I0, I1, 1.0)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n//2)
        second_half = self.make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def infer(self):
        videogen = []
        for f in os.listdir(self.img):
            if 'png' in f:
                videogen.append(f)
        videogen = videogen[0:self.maxLen]
        tot_frame = len(videogen)
        videogen.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
        lastframe = cv2.imread(os.path.join(self.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()

        videogen = videogen[1:]
        h, w, _ = lastframe.shape
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        tmp = max(32, int(32 / 1.0))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=tot_frame)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_read_buffer, (read_buffer, videogen))

        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = F.pad(I1, padding)
        temp = None # save lastframe when processing static frame
        # inference
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1.to(self.device)
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.

            I1 = F.pad(I1, padding)
            I0_small = F.interpolate(I0, (32, 32), mode='nearest-exact')
            I1_small = F.interpolate(I1, (32, 32), mode='nearest-exact')
            # Fix: 
            # This function is not deterministic. A solution is required here.
            ssim = ssim_matlab(I0_small[:, :3].cpu(), I1_small[:, :3].cpu())
            break_flag = False
            if ssim > 0.996:        
                frame = read_buffer.get() # read a new frame
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = F.pad(I1, padding)
                I1 = self.model.inference(I0, I1, 1.0)
                I1_small = F.interpolate(I1, (32, 32), mode='nearest-exact').to(self.device)
                # Fix: 
                # This function is not deterministic. A solution is required here.
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            if ssim < 0.2:
                output = []
                for i in range((2 ** self.exp) - 1):
                    output.append(I0)
            else:
                output = self.make_inference(I0, I1, 2**self.exp-1) if self.exp else []
            self.saveImg(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                self.saveImg(mid[:h, :w])
            pbar.update(1)
            lastframe = frame
            if break_flag:
                break
        self.saveImg(lastframe)
        pbar.update(1)
        pbar.close()

