import torch
from PIL import Image
from torchvision import transforms
import os
from config import cfg
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from typing import Union
import cv2

def min_max_norm(x):
    # x is a numpy array
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def preprocess_img(path_to_img):
    """
    Preprocess an image to be fed to DiNOv2
    """

    img = Image.open(path_to_img)
    img = transforms.Resize(500, antialias=True)(img)
    img = transforms.RandomAutocontrast(p=1)(img)

    h, w = img.size
    new_w = w - (w % 14)
    new_h = h - (h % 14)


    img = transforms.ToTensor()(img)

    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(img) 

    img = transforms.Resize((new_w, new_h), antialias=True)(img)

    img = img.unsqueeze(0).to(device)

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
                               threshold: Union[float,str] = 0.5, # float or "otsu"
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

    conved = min_max_norm(conved)

    if threshold == "otsu":
        thresholded = cv2.threshold(
            (255*conved).astype(np.uint8),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]/255 # second element of the tuple is the thresholded image
    else: 
        thresholded = conved > threshold

    return thresholded




if __name__ == "__main__":

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    #root / folder / img_name

    root = cfg.dataset.img_dir
    folder = os.listdir(root)[45]

    img_name = os.listdir(os.path.join(root, folder))[1]

    path_to_img = os.path.join(root, folder, img_name)

    img = Image.open(path_to_img)

    h, w = img.size

    new_w = w - (w % 14)
    new_h = h - (h % 14)

    img = transforms.ToTensor()(img)

    img = transforms.Resize((new_w, new_h), antialias=True)(img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    img = img.to(device).unsqueeze(0)

    attn = get_self_attention(model, img)

    attn = postprocess_attention_maps(attn, threshold="otsu")

    plt.imshow(attn)
    plt.axis("off")
    plt.show()



