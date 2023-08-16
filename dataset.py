from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os

from torchvision import transforms
from custom_transform import LocalizedRandomResizedCrop

from config import cfg
import tqdm as tqdm

import random as rd
import torchvision

from PIL import Image

class CUBDataset(Dataset):

    def __init__(self,
                 split : str, # train or val

                 path_to_img, 
                 img_names, 
                 indexes, 
                 label_file, 
                 box_file, 

                 
                 use_box = False, 
                 patch_size = 14, 
                 size = 224, 
                 THR = 0.7,
                 pre_crop_transforms = transforms.Compose([]),
                 post_crop_transforms = transforms.Compose([]),
                 ) -> None:
        super().__init__()

        self.split = split

        self.path_to_img = path_to_img
        self.img_names = img_names
        self.indexes = indexes
        self.label_file = label_file
        self.box_file = box_file

        self.use_box = use_box
        self.patch_size = patch_size
        self.size = size
        self.THR = THR
        
        self.pre_crop_transforms = pre_crop_transforms
        self.post_crop_transforms = post_crop_transforms
        
        if self.split == "train":

            print("Train")
            if THR < 0:
                self.use_box = False
                print("Negative threshold, using random crop")
                # using random crop when THR < 0, THR can be 0
            if self.use_box:
                print(f"Using localized crop with threshold {self.THR}")
            else:
                print("Using random crop")
        
    
        elif self.split == "val":
            print("Val")
            assert self.use_box == False, "Cannot use box in val"
  
        else:
            raise ValueError(f"split must be either train or val, got {self.split}")


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

        # Open everything
        id = self.indexes[index]

        img_name = os.path.join(self.path_to_img, self.img_names[id]) # img_names[index] contains the folder
        label = self.labels[id]

        img = Image.open(img_name).convert("RGB")
        

        # Preprocessing
        img = self.pre_crop_transforms(img) # mostly color augmentation, done before crop, these transforms will not mess up the box


        # Crops

        if self.split == "train":

            if self.use_box:
                box = self.boxes[id]
                img = LocalizedRandomResizedCrop(img, *box, size = self.size, patch_size = self.patch_size, THR=self.THR)
                
            else:
                img = transforms.RandomResizedCrop((self.size,self.size))(img)

        else:
            img = transforms.Resize(self.size)(img)
            img = transforms.CenterCrop(self.size)(img)


        # Postprocessing

        img = self.post_crop_transforms(img).float()

        label = torch.tensor(label).long()

        return img, label # img is a tensor, label is an int
 
from custom_transform import get_non_geo_transforms

def load_cub_datasets(cfg, all_vanilla = False):
    '''
    DON'T CHANGE THE RATIO
    '''
    ratio = 0.7


    # Augmentations

    if cfg.augment: # color augmentations
        pre_crop_transforms = transforms.RandomChoice(get_non_geo_transforms(p=cfg.augment_p))
    else:
        pre_crop_transforms = transforms.Compose([])


    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]) # imagenet stats

    if cfg.use_augment_mix:
        post_crop_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=2),
            transforms.ToTensor(),
            normalize
            ])
    else:
        post_crop_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

    
    img_names = {}

    with open(cfg.dataset.img_file,"r") as f:
        temp = f.readlines()
        # format i path
        img_names = {int(line.split()[0]):line.split()[1] for line in temp} # index : path
        

    list_indexes = list(img_names.keys()) # basically a range(0,11788) but ensures that all indexes are present

    rd.seed(42) # hardcoded seed to prevent mixing train and val when transfering from a pretrained model
    rd.shuffle(list_indexes)

    train_indexes = list_indexes[:int(len(list_indexes)*ratio)]
    val_indexes = list_indexes[int(len(list_indexes)*ratio):]

    if all_vanilla:
        post_crop_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ]) 
        pre_crop_transforms = transforms.Compose([])
        print("Using vanilla dataset, the train dataset will behave like the val dataset")

    train = CUBDataset(split="train" if not all_vanilla else "val",
                       path_to_img=cfg.dataset.img_dir,
                       img_names = {i:img_names[i] for i in train_indexes},
                       indexes = train_indexes,
                       label_file=cfg.dataset.label_file,
                       box_file=cfg.dataset.box_file,
                       use_box = cfg.use_box,
                       size = cfg.img_size,
                       patch_size = cfg.patch_size,
                       THR = cfg.THR,
                       
                       pre_crop_transforms= pre_crop_transforms,
                       post_crop_transforms= post_crop_transforms,

                        )
    
    val = CUBDataset(split="val",
                     path_to_img=cfg.dataset.img_dir,
                     img_names = {i:img_names[i] for i in val_indexes},
                     indexes = val_indexes,
                     label_file=cfg.dataset.label_file,
                     box_file=cfg.dataset.box_file,

                     use_box = False, # no box in val
                     size = cfg.img_size,
                     patch_size = cfg.patch_size,
                     THR = 0,

                     pre_crop_transforms= transforms.Compose([]), # no color augmentations
                     post_crop_transforms= transforms.Compose([transforms.ToTensor(),normalize]), # classic post crop transforms
                    
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
                        
    









