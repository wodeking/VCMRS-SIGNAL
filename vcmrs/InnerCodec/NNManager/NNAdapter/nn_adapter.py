# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
from distutils.command.config import config
import torch
from e2evc.Utils import utils as e2evcutils
from vcmrs.Utils.data_utils import get_frame_from_raw_video, cvt_yuv444p10b_to_bgr, limited_to_full
from .nn_utils import  pad_tensor, remove_padding
from .models import Autoencoder
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import vcmrs

class NNAdapter:
    def __init__(self, config, device="cuda"):
    
        self.config = config
        self.device = device
        self._init_params()

    def _init_params(self):
        self.model = Autoencoder(
                    en_layer_channels=self.config.en_layer_channels,
                    en_strides=self.config.en_strides,
                    resblocks_size=self.config.resblocks_size,
                    skip_connection=self.config.skip_connection,
                    lateral_connection=self.config.lateral_connection,
                    injections=self.config.injections,
                    inj_out_chn=self.config.inj_out_chn,
                    inj_operator=self.config.inj_operator).to(self.device)

        # todo: 
        vcmrs.debug(f"Number of parameters in the adapter: {e2evcutils.count_parameters(self.model):,}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        vcmrs.log("Loaded adapter model's states.")

    def get_input_data(self, img, resolution, frame_qp, sn=False):
        """
        Handles the injections and adding of the QP or resolution to the adapter input
        """
        H,W=resolution #resolution normalized (min (416 x 240)px and max (3840 × 2176)px )
        resolution = (H*W - 99840) / (8355840 - 99840)
        
        if sn: #sn stands for special normalization of the QP values
            frame_qp = (int(frame_qp) - 37) / 10
        else:
            frame_qp = int(frame_qp) / 63 # QP normalized (0-63)

        if self.config.en_layer_channels[0]==4:
            qp_tens=torch.full((1, H, W), frame_qp).type(torch.float32).to(self.device)
            img = torch.cat((img, qp_tens),0)
        elif self.config.en_layer_channels[0]==5:
            qp_tens=torch.full((1, H, W), frame_qp).type(torch.float32).to(self.device)
            reso_tens=torch.full((1, H, W), resolution).type(torch.float32).to(self.device)
            img = torch.cat((img, qp_tens, reso_tens),0)
        if self.config.injections==[1,1]:
            injection_data = torch.tensor((frame_qp, resolution), dtype=torch.float32).unsqueeze(0).to(self.device)
        elif self.config.injections==[1,0]:
            injection_data = torch.tensor((frame_qp), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            injection_data = None
        return img.unsqueeze(0), injection_data

    def apply_adapter_yuv(self, input_fp, video_info, param):
        yuv_fname, frame_idx = input_fp.split(':')
        frame_idx = int(frame_idx)

        H, W, C = video_info.resolution
        img_yuv = get_frame_from_raw_video(yuv_fname, frame_idx, 
            W = W, H = H,
            chroma_format = video_info.chroma_format,
            bit_depth=video_info.bit_depth)
        #img = limited_to_full(img_yuv)
        img = img_yuv
        img = cvt_yuv444p10b_to_bgr(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ToTensor()(img).to(self.device)
        
        frame_qp=param.qp
        img, injection_data = self.get_input_data(img, (H,W), frame_qp, sn=False)
        img = pad_tensor(img, 8, 'replicate')
        adapted_img = self.model(img, injection_data).squeeze()
        adapted_img = remove_padding(adapted_img, (H,W))
        return adapted_img

    def apply_adapter_rgb(self, input_fp, frame_qp, original_size):
        img = e2evcutils.load_img_tensor(input_fp) # NCHW, in [0,1]
        N,C,H,W = img.shape
        img = pad_tensor(img, 8, 'replicate')

        img = img.to(self.device)
        img, injection_data = self.get_input_data(img.squeeze(), (H,W), frame_qp, sn=True)
        adapted_img = self.model(img, injection_data).squeeze().clamp(0, 1)
        adapted_img = remove_padding(adapted_img, (H,W))
        return adapted_img
 
   
