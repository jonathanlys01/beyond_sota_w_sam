import torch
from torch import nn


load_refs = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14"
             ]

repo_ref = "facebookresearch/dinov2"

def get_model(type, MLP_dim, n_classes):
    """
    type: str -> model type: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14 (changes the model size)
    MLP_dim: int -> dimension of the MLP head
    n_classes: int -> number of classes

    returns: torch.nn.Module -> the model with the head replaced

    The architecutre of the head is:
    nn.Linear(model.head.in_features, MLP_dim),
    nn.ReLU(),
    nn.Linear(MLP_dim, n_classes),
    nn.Softmax(dim=1)
    """
    if not type in load_refs:
        raise ValueError("Invalid model type, should be in {}".format(load_refs))
    
    model = torch.hub.load(repo_ref, type, pretrained=True)
    

    model.head = nn.Sequential(
        nn.Linear(model.norm.weight.shape[0], MLP_dim), 
        nn.BatchNorm1d(MLP_dim),
        nn.ReLU(),
        nn.Linear(MLP_dim, n_classes),
        nn.Softmax(dim=-1)
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    for param in model.norm.parameters():
        param.requires_grad = True

    model.head.train()
    model.norm.train()

    return model