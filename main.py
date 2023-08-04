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

    for iter in range(cfg.num_epochs):

        for i,(images,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            if i%32 == 0:
                ema.update_parameters(model)
        
        scheduler.step()

        if (iter+1)%cfg.log_interval == 0:
            acc, loss_ = val_model(model,criterion,val_loader)

            ema_acc, ema_loss = val_model(ema,criterion,val_loader)
            
            if cfg.wandb:

                wandb.log({"loss":loss_,
                            "acc":acc,})
                
                wandb.log({"ema_loss":ema_loss,
                            "ema_acc":ema_acc,})
                
            
            print(f"Epoch [{iter+1}/{cfg.num_epochs}] | Loss: {loss_.item():.4f} | Acc: {acc:.4f} (ema : {ema_acc:.4f}))")

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


    
def main(cfg, project = "beyond_sota", name = None):

    if cfg.wandb:
        run = wandb.init(project=project,
                         config=cfg,
                         name=name,)

    train_loader,val_loader = load_cub_datasets(cfg)


    if cfg.model.type == "resnet50":
        model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    
        model.fc = nn.Linear(model.fc.in_features,cfg.model.n_classes)
        for m in model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif cfg.model.type.startswith("dinov2"):
        model = get_model(cfg.model.type,cfg.model.n_classes)
    else:
        # TODO: add other models
        raise NotImplementedError 


    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ema = ExponentialMovingAverage(model, decay=cfg.other.ema_decay)

    ema.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    for param in model.parameters():
        param.requires_grad = True
    params = model.parameters()
    

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

    train_model(model,ema, criterion,optimizer, scheduler ,train_loader, val_loader, cfg)

    final_acc, _ = val_model(ema,criterion,val_loader)

    final_acc = final_acc.item()

    print(f"Final accuracy : {round(final_acc*100,3)}%")

    if cfg.save:
        if name is None:
            name = f"{cfg.model.type}_{cfg.num_epochs}ep_"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.mkdir("checkpoints", exist_ok=True)
        torch.save(ema.state_dict(),f"{name}ep_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")
    if cfg.wandb:
        wandb.finish()
    

import socket
if __name__=="__main__":

    machine_name = socket.gethostname()

    print(f"Running on {machine_name}")


    cfg.use_box = False
    main(cfg,name="resnet50-cub_no_box")

    cfg.use_box = True
    cfg.alpha = 0.1
    main(cfg,name="resnet50-cub_box_alpha_0.1")

    cfg.alpha = 0.5
    main(cfg,name="resnet50-cub_box_alpha_0.5")

    cfg.alpha = 1
    main(cfg,name="resnet50-cub_box_alpha_1.0")





    """"while (ans:= input("Do you want to save the model? (y/n) ")).lower() not in ["y","n"]:
        print("Invalid input, please try again!")


    if ans.lower() == "y":
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.mkdir("checkpoints", exist_ok=True)
        torch.save(ema.state_dict(),f"{cfg.model.type}_{cfg.num_epoch}ep_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")

    elif ans.lower() == "n":
        print("Model not saved! Continuing...")"""