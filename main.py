import torch
import torchvision

from dataset import load_datasets

from config import cfg
from model import FrozenVit

from tqdm import tqdm

def train_model(model : FrozenVit, 
                criterion,
                optimizer,
                train_loader,
                cfg,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model.model.heads.train()

    for i in range(cfg.num_epochs):

        for i,(images,labels) in enumerate(tqdm(train_loader),total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            if (i+1)%cfg.train_log_interval == 0:
                print(f"Epoch {i+1}/{cfg.num_epochs}, Loss: {loss.item():.4f}")

def val_model(model : FrozenVit,
              criterion,
              val_loader,
              cfg,
              ):
    model.model.heads.eval()
    # define the accuracy metric
    

