# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import math
import numpy as np
from torchvision.transforms import RandomCrop, Resize
import cv2
from PIL import Image

# get rescaling transforms
def get_rescaling(opt):
  norm_method = getattr(opt, 'normalize', 'scale')
  if norm_method in ['scale', 'standard']:
    rescaling = lambda x: (x-0.5) * 2 # from [0, 1] to [-1, 1]
  elif norm_method == 'cifar':
    rescaling = Normalize(
      mean=(0.4914, 0.4822, 0.4465),
      std=(0.2471, 0.2435, 0.2616))
  else:
    rescaling = lambda x: x
  return rescaling

# first crop the image with a scaled croped size, then resize the cropped image to 
# the desired size
class RandomScaleCrop(object):
    def __init__(self, size, scales=[1,2]):
      self.size=size
      self.scales=scales

    def __call__(self, img):
        """
        :param img: (PIL): Image 

        :return: ycbr color space image (PIL)
        """
        crop_size = np.random.choice(self.scales)*self.size
        img = RandomCrop(crop_size, pad_if_needed=True)(img)
        img = Resize(self.size)(img)
        return img

    def __repr__(self):
        return self.__class__.__name__+'()'

def batch_images_fn(images, size_divisible=1):
  """
      Batch the given image list `images`, pad if needed
      Params:
      ----
      - `images`: the list of images (can be in different sizes)
      - `size_divisible`: the output size of the batch is divisible by this number
  """
    # concatenate
  max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

  stride = size_divisible
  max_size = get_padded_size(max_size, stride)

  batch_shape = (len(images),) + max_size
  batched_imgs = images[0].new(*batch_shape).zero_()
  for img, pad_img in zip(images, batched_imgs):
      pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

  return batched_imgs

def get_padded_size(ori_shape, stride):
  # Expect ori_shape to be in NCHW or CHW
  padded_shape = list(ori_shape)
  padded_shape[-1] = int(math.ceil(float(padded_shape[-1]) / stride) * stride)
  padded_shape[-2] = int(math.ceil(float(padded_shape[-2]) / stride) * stride)
  padded_shape = tuple(padded_shape)
  return padded_shape


# color conversion transforms
class RGB2YUV420():
  """Convert image color space from RGB to YUV
  """

  def __init__(self):
    pass


  def __call__(self, sample):
    img = sample

    img = np.array(img)
    img_yuv444 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    H,W,C = img.shape
    
    Y,U,V = img_yuv444.transpose(2,0,1)
    
    # downsample U and V
    u2 = cv2.resize(U, (W//2, H//2), cv2.INTER_AREA)
    v2 = cv2.resize(V, (W//2, H//2), cv2.INTER_AREA)
    
    # upsample it back
    UU = cv2.resize(u2, (W, H), cv2.INTER_NEAREST)
    VV = cv2.resize(v2, (W, H), cv2.INTER_NEAREST)
    
    img_yuv420 = np.stack([Y, UU, VV]).transpose(1,2,0)
    img_pil = Image.fromarray(img_yuv420)
    return img_pil

