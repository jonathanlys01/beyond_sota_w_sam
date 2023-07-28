from segment_anything import sam_model_registry, SamAutomaticMaskGenerator # TODO resolve imports later
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

in_path = "path/to/input"

out_path = "path/to/output"


for img_name in tqdm(os.listdir(in_path)):
    img = cv2.imread(os.path.join(in_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img)
    
    L_mask = []
    for i, mask in enumerate(masks):
        matrix = mask["segmentation"]
        L_mask.append(matrix)

    L_mask = np.array(L_mask)

    torch.save(torch.from_numpy(L_mask), os.path.join(out_path, img_name.split(".")[0] + ".pt"))



print("Done!")


    

    
    




