import torch
from torch import nn


load_refs = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14"
             ]

repo_ref = "facebookresearch/dinov2"

def get_model(type, n_classes):
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
    

    model.head = nn.Linear(model.norm.weight.shape[0], n_classes)
        

    for param in model.parameters():
        param.requires_grad = True

    model.train()

    return model



class NCM(nn.Module):
    """
    NCM class for the nearest class mean classifier, susing the cosine similarity as distance metric
    """
    def __init__(self, n_classes, feat_dim):
        """
        n_classes: int -> number of classes
        feat_dim: int -> dimension of the features (before the head)
        """
        super(NCM, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.zeros(n_classes, feat_dim)    ) 
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: torch.Tensor -> features before the head

        returns: torch.Tensor -> the logits of the NCM classifier
        
        """

        # B x n_feats -> B x n_classes


        return nn.Softmax(dim=-1)(self.cos(x.unsqueeze(1), self.centers.unsqueeze(0)))