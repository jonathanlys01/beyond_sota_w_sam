import torch
import torchvision
import wandb

wandb.login()

from dataset import load_cub_datasets

from config import cfg

from tqdm import tqdm
import torch.nn as nn
from model import get_model
import datetime
import os
from utils import ExponentialMovingAverage

def train_model(model : nn.Module ,
                ema : nn.Module, 
                criterion,
                optimizer,
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
    
        for i,(images,labels) in tqdm(enumerate(val_loader),total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()

            loss_t += loss.item()
            acc_t += acc.item()

    return acc_t/len(val_loader), loss_t/len(val_loader)
    
def main(cfg, project = "beyond_sota", name = None):

    if cfg.wandb:
        run = wandb.init(project=project,
                         config=cfg,
                         name=name,)

    train_loader,val_loader = load_cub_datasets(cfg)
            
    #model = get_model(cfg.model.type,cfg.model.MLP_dim,cfg.model.n_classes)

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features,cfg.model.n_classes),
                             nn.Softmax(dim=1))

    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ema = ExponentialMovingAverage(model, decay=cfg.other.ema_decay)

    ema.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    
    #params = []
    #params.extend(model.head.parameters())

    """params = [
        {'params':model.conv1.parameters()},
        {'params':model.bn1.parameters()},
        {'params':model.relu.parameters()},
        {'params':model.maxpool.parameters()},
        {"params":model.layer1.parameters()},
        {"params":model.layer2.parameters()},
        {"params":model.layer3.parameters()},
        {"params":model.layer4.parameters()},
        {"params":model.avgpool.parameters()},
        {"params":model.fc.parameters(), "lr":cfg.opt.lr*100}, # last layer with higher lr
    ]"""
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.fc.parameters():
        param.requires_grad = True

    params = model.fc.parameters()
    

    if cfg.opt.type == "adam":
        optimizer = torch.optim.Adam(params,lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "sgd":
        optimizer = torch.optim.SGD(params,lr=cfg.opt.lr, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "adamW":
        optimizer = torch.optim.AdamW(params,lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)



    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.other.label_smoothing)

    print("First training")
    cfg.num_epochs = 100
    train_model(model,ema, criterion,optimizer,train_loader, val_loader, cfg)
    
    print("Second training")
    cfg.num_epochs = 200
    for param in model.parameters():
        param.requires_grad = True


    params = [
        {'params':model.conv1.parameters()},
        {'params':model.bn1.parameters()},
        {'params':model.relu.parameters()},
        {'params':model.maxpool.parameters()},
        {"params":model.layer1.parameters()},
        {"params":model.layer2.parameters()},
        {"params":model.layer3.parameters()},
        {"params":model.layer4.parameters()},
        {"params":model.avgpool.parameters()},
        {"params":model.fc.parameters(), "lr":cfg.opt.lr}, 
    ]

    if cfg.opt.type == "adam":
        optimizer2 = torch.optim.Adam(params,lr=cfg.opt.lr/10, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "sgd":
        optimizer2 = torch.optim.SGD(params,lr=cfg.opt.lr/10, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.type == "adamW":
        optimizer2 = torch.optim.AdamW(params,lr=cfg.opt.lr/10, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
    
    train_model(model,ema, criterion,optimizer2,train_loader, val_loader, cfg)

    final_acc, _ = val_model(model,criterion,val_loader)
    
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("models",exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}_{final_acc:.4f}_{date}.pth")


if __name__=="__main__":

    cfg.use_box = False
    main(cfg,name="dino-sota-cub_base")

    cfg.use_box = True
    cfg.alpha = 0.3
    main(cfg,name="dino-sota-cub_box03")

    cfg.use_box = True
    cfg.alpha = 0.5
    main(cfg,name="dino-sota-cub_box05")

    wandb.finish()
    


    """while (ans:= input("Do you want to save the model? (y/n) ")).lower() not in ["y","n"]:
        print("Invalid input, please try again!")


    if ans.lower() == "y":
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.mkdir("checkpoints")
        torch.save(model.head.state_dict(),f"{cfg.model.type}_{cfg.num_epoch}ep_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")

    elif ans.lower() == "n":
        print("Model not saved!")"""

            
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
