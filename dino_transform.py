import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T

from typing import Optional, Union, Sequence, List, Tuple
import warnings
import math
from torch import Tensor



from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import cv2


class DiNORandomResizedCrop(torch.nn.Module):
      
    """
    Similar to the random resized crop of torchvision, but the distribution 
    of the crop's center is DiNOv2's self attention distribution.
    """

    def __init__(
    self,
    size,
    dino_type: str,
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    interpolation=F.InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()

        self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio


        ############################## Load DiNOv2 model ##############################################
        dino_types = ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']
        assert dino_type in dino_types, f"dino_type should be one of {dino_types}, got {dino_type}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading DiNOv2 model {dino_type} on device {device}")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_type, pretrained=True).to(device)

        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        self.dino_model.eval()

        ##############################################################################################

    @staticmethod
    def get_params(img: Tensor, model: torch.nn.Module, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """

        
        _, height, width = F.get_dimensions(img)

        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))



            """if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w"""
            
            img = preprocess_img(img)

            attn = get_self_attention(model, img)

            attn = postprocess_attention_maps(attn)


            
            
            

            


        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string





def min_max_norm(x):
    # x is a numpy array
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def preprocess_img(img):
    """
    Preprocess an image to be fed to DiNOv2
    """
    img = T.Resize(500, antialias=True)(img)
    img = T.RandomAutocontrast(p=1)(img)

    h, w = img.size
    new_w = w - (w % 14)
    new_h = h - (h % 14)


    img = T.ToTensor()(img)

    img = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(img) 

    img = T.Resize((new_w, new_h), antialias=True)(img)

    img = img.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return img

def get_self_attention(model,imgs):

    """
    Only works for ViT models (especially DINO, DINOv2)
    
    Args:
        model: A ViT model
        imgs: A tensor of shape (B, 3, H, W) B = 1 
    Returns:
        A tensor of shape (B, num_heads, H_new, W_new) where H_new and W_new are the height and width of the attention map (size of the img divided by the patch size)
    """

    B, C, H, W = imgs.shape

    assert len(imgs.shape) == 4, "imgs must be a 4D tensor"
    assert imgs.shape[1] == 3, "imgs must have 3 channels"
    assert imgs.shape[2] % model.patch_size == 0 and imgs.shape[3] % model.patch_size == 0, "sides must be divisible by 14"

    w_featmap = W // 14
    h_featmap = H // 14

    with torch.no_grad():
        output = model.get_intermediate_layers(x=imgs,
                                            reshape=True,
                                            n = 2,
                                            return_class_token=True,
                                            )

        # output is a list of tuples (maps, class_token), the length of the list is the number of layers considered


        maps = output[0][0] 

        B, C = output[0][1].shape

        # reshape maps to be (B, N, C) where N is the number of patches

        maps = maps.reshape((B,maps.shape[1],-1)).permute(0,2,1)


        class_token = output[0][1].reshape((B,-1,1)).permute(0,2,1)


        maps = torch.cat((class_token, maps), dim=1)

        # get the last attention block (only qkv)with 
    
        qkv = model.blocks[-1].attn.qkv


        B, N, C = maps.shape

        qkv_out = qkv(maps).reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)


        head_dim = C // model.num_heads
        scale = head_dim**-0.5

        q, k = qkv_out[0] * scale, qkv_out[1]

        attn = q @ k.transpose(-2, -1) # (B, nh, N, N)
        
        nh = model.num_heads
        assert B == 1, "B must be 1"
        attn = attn[:, :, 0, 1:].reshape(nh, h_featmap, w_featmap)
        
        return attn.cpu().numpy()


def postprocess_attention_maps(maps: np.ndarray,
                               kernel_size: int = 3,
                               temperature: float = 2.,
                               ) -> np.ndarray:
    """
    Cleans the attention maps by applying a threshold and a gaussian filter, and averaging the attention maps over the heads
    Args:
        maps: A numpy array of shape (num_heads, H, W)
        kernel_size: The size of the gaussian kernel
        temperature: The temperature used to apply the softmax
        threshold: The threshold applied to the attention maps
    Returns:
        A numpy array of shape (H, W)
    
    """
    softmax = lambda x: np.exp(x/temperature) / np.sum(np.exp(x/temperature))

    maps = np.mean(maps, axis=0) # average over the heads

    conved = convolve2d(
        maps,
        np.ones(
                tuple(max(kernel_size,int(0.1*side)) for side in maps.shape),
                ),
        mode="same",
        boundary="symm",)
    

    conved = min_max_norm(conved)

    conved = softmax(conved)


