from functools import partial
from typing import Callable, List, Optional
import torch
import torch.nn as nn
import torchvision
from torch import nn

from torchvision.models.vision_transformer import ConvStemConfig, VisionTransformer



class FrozenVit(nn.Module):
    def __init__(self, num_classes,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        print("Loading ViT")
        self.model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        print("Weights loaded")

        self.model.heads = nn.Sequential(
            nn.Linear(self.model.hidden_dim, self.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.model.hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    

"""class MyModel(nn.Module):
    def __init__(self,fc_shape):
        super().__init__(fc_shape)
        self.fc_shape = fc_shape

        self.fc = nn.ModuleList(
            [nn.Linear(fc_shape[i],fc_shape[i+1]) for i in range(len(fc_shape)-1)]
        )
        self.act = nn.ModuleList(
            [nn.ReLU() for i in range(len(fc_shape)-1)]
        )

        self.softmax = nn.Softmax(dim=1)

        self.model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self.model.

        

    def forward(self,x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        #x = self.model.heads(x)

        for i in range(len(self.fc)):
            x = self.fc[i](x)
            x = self.act[i](x)
        
        x = self.softmax(x)


        return x"""

        
        