from copy import deepcopy
from typing import Any
import torch.nn as nn
from config import cfg
import torchvision
import torch
from torchvision.models import ResNet
import os
from tqdm import tqdm
import numpy as np
import json
import argparse

from dataset import load_box_dataset


"""
Generates the bounding boxes from the model features and the crops from SAM's masks
"""

def get_copy_features(model : ResNet, type_ = 'resnet') -> ResNet:
    copy = deepcopy(model)
    
    if type_.startswith('resnet'):
        copy.fc = nn.Identity()
    
    elif type_.startswith("dino"):
        copy.head = nn.Identity()
    
    return copy


argparser = argparse.ArgumentParser()

argparser.add_argument("--name", "-n",type=str, default="masks_0", help="masks folder name")

name = argparser.parse_args().name

seg = os.path.join(
    os.path.dirname(cfg.dataset.img_dir), # parent directory
    name
)

assert os.path.exists(seg)

seg_file = os.path.join(seg,
                       "masks_info.json")

with open(seg_file) as f:
    seg_info = json.load(f)
print(seg_info["setup"])
del seg_info


cfg.batch_size = 32
train_dataset, val_dataset = load_box_dataset(cfg, seg_file)


model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, cfg.model.n_classes)
model.load_state_dict(torch.load(cfg.model.resumed_model))

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

for i, (imgs,_, labels, boxes, gts) in tqdm(enumerate(train_dataset), total=len(train_dataset)): # _ is the img_index

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
for imgs,idxs,labels, boxes, gts in tqdm(val_dataset, total=len(val_dataset)):

    imgs = imgs.to(device)

    with torch.inference_mode():
        outs = model_features(imgs)
    # batch inference -> unbatch

    for out, idx, label, box, gt in zip(outs, idxs, labels, boxes, gts):

        out = out.unsqueeze(0) # size (1, 2048)
    
        logits = cosine(out, features_list).cpu().numpy() # size (n_classes,)

        if (not (int(idx) in d.keys())):

            if label == np.argmax(logits)+1:

                d[int(idx)] = {"label": int(label), 
                            "logits": logits,
                                "box": box.tolist(),
                                "gt": gt.tolist() 
                            }
        
        else:

            if max(d[int(idx)]["logits"]) < max(logits) and label == d[int(idx)]["label"]:
                    # replace the mask the new logits are better
                    d[int(idx)] = {"label": int(label), 
                                   "logits": logits,
                                    "box": box.tolist(),
                                    "gt": gt.tolist()
                                   }
            
acc = sum(
        [1 for k,v in d.items() if v["label"] == np.argmax(v["logits"])] # if the label is the same as the argmax of the logits
        )

print(f"Accuracy: {100*acc/len(d)} ({acc}/{len(d)})")


def iou(gt,bbox, mode="std"):
    """
    gt and bbox :  x,y, w, h
    """
    x1, y1, w1, h1 = gt
    x2, y2, w2, h2 = bbox

    area1 = w1*h1
    area2 = w2*h2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1, x2+w2)
    yB = min(y1+h1, y2+h2)
    
    interArea = max(0, xB-xA) * max(0, yB-yA)
    if mode == "std":
        unionArea = area1 + area2 - interArea
    elif mode == "as-gt":
        unionArea = area1
    elif mode == "as-bbox":
        unionArea = area2
    else:
        raise ValueError("mode must be one of 'std', 'as-gt', 'as-bbox'")

    eps = 1e-6

    iou = interArea / max(unionArea,eps)

    return iou


miou = (sum([iou(v['gt'],v['box']) for k,v in d.items()])/len(d))

print(f"mIOU : {miou}")

miougt = (sum([iou(v['gt'],v['box'], mode='as-gt') for k,v in d.items()])/len(d))

print(f"mIOU as gt : {miougt}")

mioubbox = (sum([iou(v['gt'],v['box'], mode='as-bbox') for k,v in d.items()])/len(d))

print(f"mIOU as bbox : {mioubbox}")



import matplotlib.pyplot as plt
plt.subplot(3,1,1)

plt.hist(
    [iou(v['gt'],v['box']) for k,v in d.items()],
    bins=100   
)

plt.title(f"IOU distribution (mIOU: {miou:.3f})")

plt.subplot(3,1,2)
plt.hist(
    np.array([iou(v['gt'],v['box'], mode='as-gt') for k,v in d.items()]),
    bins=100
)

plt.title(f"IOU distribution as gt (mIOU: {miougt:.3f})")

plt.subplot(3,1,3)

plt.hist(
    np.array([iou(v['gt'],v['box'], mode='as-bbox') for k,v in d.items()]),   
    bins=100
)

plt.title(f"IOU distribution as bbox (mIOU: {mioubbox:.3f})")

plt.show()

