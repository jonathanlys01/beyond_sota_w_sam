import torch
import torchvision
import wandb



from dataset import load_cub_datasets

from config import cfg

if cfg.wandb:
    wandb.login()

from tqdm import tqdm
import torch.nn as nn
from model import get_model
import datetime
import os
from utils import ExponentialMovingAverage

from model import NCM

def train_model(model : nn.Module ,
                ema : nn.Module, 
                criterion,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                cfg,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model.train()

    for epoch in range(cfg.num_epochs):

        for i,(images,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            if i%cfg.model.ema_step == 0:
                ema.update_parameters(model)
        
        scheduler.step()

        if (epoch+1)%cfg.log_interval == 0:
            acc, loss_ = val_model(model,criterion,val_loader)

            ema_acc, ema_loss = val_model(ema,criterion,val_loader)

            lr=scheduler.get_last_lr()[0]
            
            if cfg.wandb:

                wandb.log({"loss":loss_,
                            "acc":acc,
                            "ema_loss":ema_loss,
                            "ema_acc":ema_acc,
                            "lr":lr,
                            "epoch":epoch+1,})

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] | Loss: {loss_.item():.4f} | Acc: {acc:.4f} (ema : {ema_acc:.4f})) | lr : {lr:.4f}")

def val_model(model : nn.Module,
              criterion,
              val_loader,
              ):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_t = torch.tensor(0.0).to(device)
    acc_t = torch.tensor(0.0).to(device)

    with torch.inference_mode():
    
        for i,(images,labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()

            loss_t += loss.item()
            acc_t += acc.item()

    return acc_t/len(val_loader), loss_t/len(val_loader)


def train_ncm(model : nn.Module,train_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.fc = nn.Identity()

    n_features = model.layer4[-1].bn3.num_features

    centers = (torch.zeros(cfg.model.n_classes, n_features)).to(device) 

    with torch.inference_mode():

        for i, (images,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            for output,label in zip(outputs,labels):
                idx = torch.argmax(label)
                centers[idx] += output

    for center in centers:
        center /= len(train_loader)*cfg.batch_size # divide by number of images

    model.fc = NCM(n_features, cfg.model.n_classes)
    model.fc.centers = torch.nn.Parameter(centers) # transpose to get n_features x n_classes


    
def main(cfg, name = None):


    train_loader,val_loader = load_cub_datasets(cfg)


    if cfg.model.type == "resnet50":
        model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    
        model.fc = nn.Linear(model.fc.in_features,cfg.model.n_classes)
        for m in model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        if cfg.model.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

    elif cfg.model.type.startswith("dinov2"):
        model = get_model(cfg.model.type,cfg.model.n_classes)
        if cfg.model.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
    else:
        # TODO: add other models
        raise NotImplementedError 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ExponentialMovingAverage(model, decay=cfg.other.ema_decay)

    ema.to(device)


    if cfg.model.resumed_model is not None:
        ema.load_state_dict(torch.load(cfg.model.resumed_model))

        model.load_state_dict(ema.module.state_dict())


    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    

    if cfg.opt.type == "adam":
        optimizer = torch.optim.Adam(params,lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "sgd":
        optimizer = torch.optim.SGD(params,lr=cfg.opt.lr, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "adamW":
        optimizer = torch.optim.AdamW(params,lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)


    if cfg.opt.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    elif cfg.opt.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.opt.step_size, gamma=cfg.opt.gamma)


    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.other.label_smoothing)

    start_acc, _ = val_model(ema,criterion,val_loader)

    start_acc = start_acc.item()

    print(f"Start accuracy (ema) : {round(start_acc*100,3)}%")

    train_model(model,ema, criterion,optimizer, scheduler ,train_loader, val_loader, cfg)

    final_acc, _ = val_model(ema,criterion,val_loader)

    final_acc = final_acc.item()

    print(f"Final accuracy (ema) : {round(final_acc*100,3)}%")

    if cfg.save:
        if name is None:
            name = f"{cfg.model.type}_{cfg.num_epochs}ep_thr{cfg.THR}"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if os.path.isdir("models") is False:
            os.mkdir("models")
        torch.save(ema.state_dict(),f"models/{name}_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")

    

def main_sweep(config=None):

    if config is not None:
        print("Current config : ",config)
        cfg.THR = config["THR"]

    with wandb.init(config=cfg):     
        main(cfg)
    wandb.finish()




############################################################################################################################################################################

import socket
import random
import numpy as np

if __name__=="__main__":

    
    if cfg.deterministic: 
        seed_ = cfg.seed
        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        torch.backends.cudnn.deterministic = True

    machine_name = socket.gethostname()
    print(f"Running on {machine_name}")

    if cfg.sweeprun: # sweep
        sweep_config = {
            'method': 'random', #grid, random
            'metric': {
                'name': 'ema_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'THR': {
                    "values": [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                },
            }}
        sweep_id = wandb.sweep(sweep_config, project="beyond_sota_sweep")

        wandb.agent(sweep_id, function=main_sweep, count=1)

    else: # single run
        cfg.use_box = False
        cfg.model.freeze_backbone = False
        cfg.model.resumed_model = 'models/resnet50-ImageNetWeights_ac57.936_2023-08-07_12:40:16.pt'
        name = "resnet50-ImageNetWeights_resume"

        if cfg.wandb:
            run = wandb.init(project="beyond_sota",
                            config=cfg,
                            name=name,)
            
        main(cfg,name=name)

        if cfg.wandb:
            wandb.finish()




    """"while (ans:= input("Do you want to save the model? (y/n) ")).lower() not in ["y","n"]:
        print("Invalid input, please try again!")


    if ans.lower() == "y":
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.mkdir("checkpoints", exist_ok=True)
        torch.save(ema.state_dict(),f"{cfg.model.type}_{cfg.num_epoch}ep_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")

    elif ans.lower() == "n":
        print("Model not saved! Continuing...")"""