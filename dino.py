"""
TODO: Replace the following code with function related to the use of DINO to extract self attention maps (relative to the cls token) from the images.
Also add the code to compute the normalized masked attention maps (as a probability of the mask being an object (wrt the cls token))
"""


import torch


from PIL import Image
from torchvision import transforms

import os

from config import cfg

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
img = img.to(device)

print(img.shape)
img = img.unsqueeze(0)
with torch.no_grad():
    out = model.forward(img,is_training=True)

for k,v in out.items():
    if v is not None:
        print(k, v.shape)

import matplotlib.pyplot as plt

att_map = out["x_norm_patchtokens"] # (B, N_patch, emb_dim)

att_map = att_map.squeeze(0)

att_map = att_map.reshape(new_w//14, new_h//14, -1)

att_map = att_map.permute(2,0,1)

# subplots

fig, axs = plt.subplots(1,2)

axs[0].imshow(img.squeeze(0).permute(1,2,0).cpu().numpy())

axs[1].imshow(torch.mea(att_map, axis=0).cpu().detach().numpy(),
              cmap='coolwarm',)


plt.show()