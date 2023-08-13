from copy import deepcopy
import torch.nn as nn
from config import cfg
import torchvision
import torch
from torchvision.models import ResNet


def get_copy_features(model : ResNet, type_ = 'resnet') -> ResNet:
    copy = deepcopy(model)
    
    if type_.startswith('resnet'):
        copy.fc = nn.Identity()
    
    elif type_.startswith("dino"):
        copy.head = nn.Identity()
    
    return copy


from dataset import load_cub_datasets

train_dataset, _ = load_cub_datasets(cfg)


features = {i: [] for i in range(cfg.num_classes)}

model = torchvision.models.resnet50(
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

model_features = get_copy_features(model)

model_features.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_features.to(device)

for i, (images, labels) in enumerate(train_dataset):
    
    images = images.to(device)
    labels = labels.to(device)
    
    features_ = model_features(images)
    
    for j, label in enumerate(labels):
        features[label.item()].append(features_[j].detach().cpu().numpy())
        
for i in features:
    features[i] = torch.tensor(features[i]).mean(dim = 0)

torch.save(features, 'features.pt') # dict of tensors of shape (feature_dim,)

    
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