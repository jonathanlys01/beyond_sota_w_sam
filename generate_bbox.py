from copy import deepcopy
import torch.nn as nn
from config import cfg
import torchvision
import torch
from torchvision.models import ResNet
import os
from tqdm import tqdm
import numpy as np

from dataset import load_box_dataset

def get_copy_features(model : ResNet, type_ = 'resnet') -> ResNet:
    copy = deepcopy(model)
    
    if type_.startswith('resnet'):
        copy.fc = nn.Identity()
    
    elif type_.startswith("dino"):
        copy.head = nn.Identity()
    
    return copy


name = "masks_0"

seg = os.path.join(
    os.path.dirname(cfg.dataset.img_dir), # parent directory
    name
)

assert os.path.exists(seg)

seg_file = os.path.join(seg,
                       "masks_info.json")


cfg.batch_size = 8
train_dataset, val_dataset = load_box_dataset(cfg, seg_file)


model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)


model_features = get_copy_features(model)

model_features.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_features.to(device)


temp_dir = os.path.join(
    os.getcwd(),
    "temp",
)

features_list = [[torch.zeros(model.fc.in_features).to(device),0] for _ in range(cfg.model.n_classes)]# classes start at 1
del model

# calculate class prototypes from training set

for i, (imgs,_, labels) in tqdm(enumerate(train_dataset), total=len(train_dataset)): # _ is the img_index

    for img, label in zip(imgs, labels):

        img = img.unsqueeze(0) # 4D tensor expected

        img = img.to(device)

        with torch.inference_mode():
            features = model_features(img)
        
        features_list[int(label)][0] += features.squeeze(0) # squeeze to remove the batch dimension
        features_list[int(label)][1] += 1

features_list = [v[0]/v[1] for v in features_list if v[1] != 0] # average the features
features_list = torch.stack(features_list).to(device) # size (n_classes, 2048)


d = {}
cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# calculate the accuracy on the validation set
for imgs,idxs,labels in tqdm(val_dataset, total=len(val_dataset)):

    imgs = imgs.to(device)

    with torch.inference_mode():
        outs = model_features(imgs)


    for out, idx, label in zip(outs, idxs, labels):

        out = out.unsqueeze(0) # size (1, 2048)
    
        logits = cosine(out, features_list).cpu().numpy() # size (n_classes,)
        if not int(idx) in d.keys():

            d[int(idx)] = {"label": int(label), 
                           "logits": logits}
        
        else:

            if max(d[int(idx)]["logits"]) < max(logits):
                    # replace the logits when the new logits are better
                    d[int(idx)] = {"label": int(label), 
                                "logits": logits}
            
print(
    sum(
        [1 for k,v in d.items() if v["label"] == np.argmax(v["logits"])] # if the label is the same as the argmax of the logits
    )
)



    
"""
THEN 
- take an image
- take its masks
- get the bbox for each mask
- get the features for each crop corresponding to the bbox
- get the crop with the maximum similarity with the features of the class
- compare this class with the class of the image to get the accuracy

- if the accuracy is good
- write the bbox in a file (txt or json)
- do the same procedure as before with the localized random resized crop

"""