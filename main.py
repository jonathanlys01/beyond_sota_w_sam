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

    L_loss = []
    L_acc = []

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
                L_acc.append(float(acc.cpu()))
                L_loss.append(float(loss_.cpu()))

                running_acc = np.mean(L_acc)
                running_loss = np.mean(L_loss)
                
                wandb.log({"loss":loss_,
                            "acc":acc,

                            "running_acc":running_acc,
                            "running_loss":running_loss,

                            "ema_loss":ema_loss,
                            "ema_acc":ema_acc,

                            "lr":lr,
                            "epoch":epoch+1,})

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] | Loss: {loss_.item():.4f} (ema : {ema_loss:.4f}) | Acc: {acc:.4f} (ema : {ema_acc:.4f})) | lr : {lr:.3g}")

def val_model(model : nn.Module,
              criterion,
              val_loader,
              ):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_t = torch.tensor(0.0).to(device)
    acc_t = torch.tensor(0.0).to(device)

    with torch.inference_mode():
    
        for i,(images,labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
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

def test_only(cfg):
    train_loader,val_loader = load_cub_datasets(cfg)
    if cfg.model.type == "resnet50":
        model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    
        model.fc = nn.Linear(model.fc.in_features,cfg.model.n_classes)
    elif cfg.model.type.startswith("dinov2"):
        model = get_model(cfg.model.type,cfg.model.n_classes)
    else:
        raise NotImplementedError
    
    ema = ExponentialMovingAverage(model, decay=0.9999)

    assert cfg.model.resumed_model is not None


    if "ema" in cfg.model.resumed_model:
        print("Resuming training from ema checkpoint")
        ema.load_state_dict(torch.load(cfg.model.resumed_model))
        model.load_state_dict(ema.module.state_dict())
    else:
        print("Resuming training from checkpoint")
        model.load_state_dict(torch.load(cfg.model.resumed_model))
        ema.module.load_state_dict(torch.load(cfg.model.resumed_model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_acc, train_loss = val_model(model,criterion,train_loader)
    print(f"Train acc : {train_acc:.4f} | Train loss : {train_loss:.4f}")


    val_acc, val_loss = val_model(model,criterion,val_loader)
    print(f"Val acc : {val_acc:.4f} | Val loss : {val_loss:.4f}")
    
def main(cfg, name = None):


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ExponentialMovingAverage(model, decay=cfg.other.ema_decay)

    ema.to(device)


    if cfg.model.resumed_model is not None:
        if "ema" in cfg.model.resumed_model:
            print("Resuming training from ema checkpoint")
            ema.load_state_dict(torch.load(cfg.model.resumed_model))
            model.load_state_dict(ema.module.state_dict())
        else:
            print("Resuming training from checkpoint")
            model.load_state_dict(torch.load(cfg.model.resumed_model))
            ema.module.load_state_dict(torch.load(cfg.model.resumed_model))

    if cfg.model.type.startswith("resnet50"):

        if cfg.model.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        nowd = []
        wd = []
        num_parameters = int(torch.tensor([x.numel() for x in model.parameters()]).sum().item())
        trained_parameters = 0

        for x in model.modules():
            if isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.Linear):
                if isinstance(x, nn.BatchNorm2d):
                    nowd.append(x.weight)
                    trained_parameters += x.weight.numel()
                else:
                    wd.append(x.weight)
                    trained_parameters += x.weight.numel()
                nowd.append(x.bias)
                trained_parameters += x.bias.numel()
            elif isinstance(x, nn.Conv2d):
                wd.append(x.weight)
                trained_parameters += x.weight.numel()
        assert(num_parameters==trained_parameters)

        print(f"Number of parameters : {num_parameters:,}")

        nowd = [param for param in nowd if param.requires_grad]
        wd = [param for param in wd if param.requires_grad]
        params = [{"params":nowd,"weight_decay":0.0},{"params":wd,"weight_decay":cfg.opt.weight_decay}]

        print(f"Parameters optimized : {sum([x.numel() for x in nowd]):,} (no weight decay) + {sum([x.numel() for x in wd]):,} (weight decay)")
    
    elif cfg.model.type.startswith("dinov2"):

        if cfg.model.train_cls:
            for param in model.parameters():
                param.requires_grad = True
            
            # fc
            wd = model.head.parameters()
            # cls
            nowd = model.cls_token # already a parameter

            params = [{"params":nowd,"weight_decay":0.0},{"params":wd,"weight_decay":cfg.opt.weight_decay}]

            print(f"Parameters optimized : {sum([x.numel() for x in nowd]):,} (no weight decay) + {sum([x.numel() for x in wd]):,} (weight decay)")

            
        else: # only train fc

            for param in model.parameters():
                param.requires_grad = False

            for param in model.head.parameters():
                param.requires_grad = True

            params = [{"params":model.head.parameters(),"weight_decay":cfg.opt.weight_decay}]

            print(f"Parameters optimized : {sum([x.numel() for x in model.head.parameters()]):,} (weight decay)")

    

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

    final_ema_acc, _ = val_model(ema,criterion,val_loader)
    final_ema_acc = final_ema_acc.item()

    final_acc, _ = val_model(model,criterion,val_loader)
    final_acc = final_acc.item()

    print(f"Final accuracy (ema) : {round(final_ema_acc*100,3)}%")

    if cfg.save:
        if name is None:
            name = f"{cfg.model.type}_{cfg.num_epochs}ep_thr{cfg.THR}"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if os.path.isdir("models") is False:
            os.mkdir("models")
        torch.save(ema.state_dict(),f"models/{name}_ema_ac{round(final_ema_acc*100,3)}_{date}.pt")
        torch.save(model.state_dict(),  f"models/{name}_ac{round(final_acc*100,3    )}_{date}.pt")
        print("Models saved!")

    
def main_sweep():


    with wandb.init(project="beyond_sota_sweep",entity="jonathanlystahiti"):

        config = wandb.config

        print("Current config : ",config)

        dummy_maiou = config["dummy_maiou"]

        
        try:
            del cfg.THR
        except:
            print("THR not in cfg")
        
        
        if 0.5 <= dummy_maiou <= 1.0:
            thr = (2*(dummy_maiou-0.5))**(1/0.4)

            maiou = dummy_maiou

        else: # 1 < dummy_maiou 
            thr = -1 # will not use box

            maiou = 1.4 + (dummy_maiou-1.0)



        cfg.THR = thr
        cfg.maIOU = maiou

        wandb.config.update(cfg)
        wandb.config["THR"] = cfg.THR
        wandb.config["maiou"] = cfg.maIOU

        print("THR : ",cfg.THR)
        print("maiou : ",cfg.maIOU)
        
        main(cfg)
    wandb.finish()


############################################################################################################################################################################

import socket
import random
import numpy as np

if __name__=="__main__":

    if cfg.model.type.startswith("dinov2"):
        assert cfg.img_size % cfg.patch_size == 0, f"Image size must be divisible by patch size, nearest size is {cfg.patch_size*round(cfg.image_size/cfg.patch_size)}"
    
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
        
        if cfg.sweep.resume:
            sweep_id = cfg.sweep.id
            print("Part of an existing sweep")
        else:
            sweep_config = {
            'method': 'grid', #grid, random
            'metric': {
                'name': 'ema_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'THR': {
                    "values": [0, 0.001, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                },
            }}
            sweep_id = wandb.sweep(sweep_config, project="beyond_sota_sweep")

        print("Sweep id : ",sweep_id)
        print("Lauching sweep")
        wandb.agent(sweep_id, function=main_sweep, count=cfg.sweep.count, 
                    entity="jonathanlystahiti", 
                    project="beyond_sota_sweep")

    else: # single run

        cfg.use_box = True
        cfg.THR = 1.0

        
        name = f"dino_box_prior"

        for name, source in zip(
            ["dino_van_run1","dino_ft_run1",
             "dino_van_run2","dino_ft_run2",],
            ["/mnt/data/CUB_200_2011/images/box_og.json",
             "/mnt/data/CUB_200_2011/images/box_ft.json",
             "/mnt/data/CUB_200_2011/images/box_og.json",
             "/mnt/data/CUB_200_2011/images/box_ft.json",
             ]
            ):

            if cfg.wandb:
                cfg.source = source
                run = wandb.init(project="beyond_sota_sweep",config=cfg,name=name,)
            
                main(cfg,name=name)
                wandb.finish()

            else:
                main(cfg,name=name)

        ############################################################################################################################################################################

