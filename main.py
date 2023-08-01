import torch
import torchvision
import wandb

from dataset import load_cub_datasets

from config import cfg

from tqdm import tqdm
import torch.nn as nn
from model import get_model
import datetime
import os



# https://docs.wandb.ai/guides/integrations/pytorch


def train_model(model : nn.Module , 
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

        if (iter+1)%cfg.log_interval == 0:
            acc, loss_ = val_model(model,criterion,val_loader)
            
            if cfg.wandb:

                wandb.log({"loss":loss_,
                            "acc":acc,})
                print("Logged")
            
            print(f"Epoch [{iter+1}/{cfg.num_epochs}] | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

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
    
def main(cfg, name = "dino-sota-cub"):

    if cfg.wandb:
        run = wandb.init(project=name,config=cfg)

    cfg.use_box = False

    train_loader,val_loader = load_cub_datasets(cfg)
            
    model = get_model(cfg.model.type,cfg.model.MLP_dim,cfg.model.n_classes)


    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    params = []
    params.extend(model.head.parameters())


    if cfg.opt.type == "adam":
        optimizer = torch.optim.Adam(params,lr=cfg.opt.lr)
    elif cfg.opt.type == "sgd":
        optimizer = torch.optim.SGD(params,lr=cfg.opt.lr)
    elif cfg.opt.type == "adamW":
        optimizer = torch.optim.AdamW(params,lr=cfg.opt.lr)



    criterion = nn.CrossEntropyLoss()

    print("Training started!")
    train_model(model,criterion,optimizer,train_loader, val_loader, cfg)
    print("Training done!")

    final_acc, final_loss = val_model(model,criterion,val_loader)
    print(f"Final accuracy: {final_acc:.4f}")


if __name__=="__main__":

    cfg.use_box = False
    main(cfg,name="dino-sota-cub_base")

    cfg.use_box = True
    cfg.alpha = 0.1
    main(cfg,name="dino-sota-cub_box01")

    cfg.use_box = True
    cfg.alpha = 0.3
    main(cfg,name="dino-sota-cub_box03")

    cfg.use_box = True
    cfg.alpha = 0.5
    main(cfg,name="dino-sota-cub_box05")

    cfg.use_box = True
    cfg.alpha = 0.7
    main(cfg,name="dino-sota-cub_box07")


    cfg.use_box = True
    cfg.alpha = 0.9
    main(cfg,name="dino-sota-cub_box09")

    cfg.use_box = True
    cfg.alpha = 0.3
    cfg.opt.type = "sgd"
    cfg.opt.lr = 0.1
    main(cfg,name="dino-sota-cub_box03_sgd")

    cfg.use_box = False
    main(cfg,name="dino-sota-cub_base_sgd")

    




    """while (ans:= input("Do you want to save the model? (y/n) ")) not in ["y","n"]:
        print("Invalid input, please try again!")


    if ans.lower() == "y":
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.mkdir("checkpoints")
        torch.save(model.head.state_dict(),f"{cfg.model.type}_{cfg.num_epoch}ep_ac{round(final_acc*100,3)}_{date}.pt")
        print("Model saved!")

    elif ans.lower() == "n":
        print("Model not saved!")"""

            
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
