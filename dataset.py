from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import cv2
from torchvision import transforms
from custom_transform import LocalizedRandomResizedCrop

from config import cfg
import tqdm as tqdm

import random as rd
import torchvision

class CUBDataset(Dataset):

    def __init__(self, path_to_img, img_names, indexes, label_file, box_file, transforms, use_box = False, patch_size = 14, size = 224, THR = 0.7) -> None:
        super().__init__()

        self.path_to_img = path_to_img
        self.img_names = img_names
        self.indexes = indexes
        self.label_file = label_file
        self.box_file = box_file
        self.use_box = use_box
        self.patch_size = patch_size
        self.size = size
        self.THR = THR

        if THR <= 0:
            self.use_box = False
            # using random crop when THR <= 0

        if self.use_box:
            print(f"Using localized crop with threshold {self.THR}")
        else:
            print("Using random crop")

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])

        self.transforms = transforms

        with open(label_file,"r") as f:
            temp = f.readlines()
            # format i class_number
            self.labels = {int(line.split()[0]) : int(line.split()[1])-1 for line in temp if int(line.split()[0]) in indexes}



        with open(box_file,"r") as f:
            temp = f.readlines()
            # format i x y w h
            self.boxes = {int(line.split()[0]):list(map(lambda x : int(float(x)), line.split()[1:])) for line in temp if int(line.split()[0]) in indexes} # remove the first element which is the image index, result in x y w h

        assert len(self.labels) == len(self.boxes) == len(self.img_names) == len(self.indexes) , "Length of labels, boxes, img_names, indexes are not equal"


    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, index):

        id = self.indexes[index]

        img_name = os.path.join(self.path_to_img, self.img_names[id]) # img_names[index] contains the folder
        label = self.labels[id]


        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # to PIL
        img = transforms.ToPILImage()(img)

        if self.use_box:
            box = self.boxes[id]
            img = LocalizedRandomResizedCrop(img, *box, size = self.size, patch_size = self.patch_size, THR=self.THR)
            
        else:
            img = transforms.RandomResizedCrop((self.size,self.size))(img)

        img = self.transforms(img).float()

        img = self.normalize(img)

        label = torch.tensor(label).long()

        return img, label # img is a tensor, label is an int
 

def load_cub_datasets(cfg, ratio = 0.7):


    img_names = {}

    with open(cfg.dataset.img_file,"r") as f:
        temp = f.readlines()
        # format i path
        img_names = {int(line.split()[0]):line.split()[1] for line in temp} # index : path
        

    list_indexes = list(img_names.keys())

    if cfg.deterministic:
        rd.seed(42) # hardcoded seed to prevent mixing train and val when transfering from a pretrained model
    rd.shuffle(list_indexes)

    train_indexes = list_indexes[:int(len(list_indexes)*ratio)]
    val_indexes = list_indexes[int(len(list_indexes)*ratio):]

    train = CUBDataset(path_to_img=cfg.dataset.img_dir,
                        img_names = {i:img_names[i] for i in train_indexes},
                       indexes = train_indexes,
                       label_file=cfg.dataset.label_file,
                       box_file=cfg.dataset.box_file,
                          transforms=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                #transforms.TrivialAugmentWide(),
                                transforms.ToTensor(),
                                #transforms.RandomErasing(0.1),
                                ]),
                          use_box = cfg.use_box,
                          size = cfg.img_size,
                          patch_size = cfg.patch_size,
                          THR = cfg.THR
                            )
    
    val = CUBDataset(path_to_img=cfg.dataset.img_dir,
                     img_names = {i:img_names[i] for i in val_indexes},
                        indexes = val_indexes,
                        label_file=cfg.dataset.label_file,
                        box_file=cfg.dataset.box_file,
                        transforms=transforms.ToTensor(),
                        use_box = False,
                        size = cfg.img_size,
                        patch_size = cfg.patch_size,
                        THR = 0
                        )
    if cfg.deterministic:
        seed = cfg.seed
        torch.manual_seed(seed)

    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=True)
    
    return train_loader, val_loader
                        
    









