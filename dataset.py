from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os

from torchvision import transforms
import torchvision.transforms.functional as F
from custom_transform import LocalizedRandomResizedCrop

from config import cfg
import tqdm as tqdm

import random as rd
import torchvision

from PIL import Image

import json

from dino_transform import DiNORandomResizedCrop

class CUBDataset(Dataset):

    def __init__(self,
                 split : str, # train or val

                 path_to_img, 
                 img_names, 
                 indexes, 
                 label_file, 
                 box_file, 

                 
                 use_box = False, 
                 use_dino = False,
                 patch_size = 14, 
                 size = 224, 
                 THR = 0.7,
                 pre_crop_transforms = transforms.Compose([]),
                 post_crop_transforms = transforms.Compose([]),

                 source = "gt", # "gt", or path to sam or dinov2_vanilla or dinov2_ft boxes
                 ) -> None:
        super().__init__()

        self.split = split

        self.path_to_img = path_to_img
        self.img_names = img_names
        self.indexes = indexes
        self.label_file = label_file
        self.box_file = box_file

        self.use_box = use_box
        self.use_dino = use_dino
        self.patch_size = patch_size
        self.size = size
        self.THR = THR
        
        self.pre_crop_transforms = pre_crop_transforms
        self.post_crop_transforms = post_crop_transforms

        self.source = source

        self.already_loaded = False
        
        if self.split == "train":

            self.dino_transform = transforms.Compose([]) 
            if THR < 0:
                self.use_box = False
                print("Negative threshold, using random crop")
                # using random crop when THR < 0, THR can be 0
            if self.use_box:
                print(f"Using localized crop with threshold {self.THR}")
            else:
                print("Using random crop")
        
            print("Using dino" if self.use_dino else "Not using dino")
    
        elif self.split == "val":
            print("Val")
            self.dino_transform = transforms.Compose([]) 
            assert self.use_box == False, "Cannot use box in val"
  
        else:
            raise ValueError(f"split must be either train or val, got {self.split}")


        with open(label_file,"r") as f:
            temp = f.readlines()
            # format i class_number
            self.labels = {int(line.split()[0]) : int(line.split()[1])-1 for line in temp if int(line.split()[0]) in indexes}

        if self.source == "gt":

            with open(box_file,"r") as f:
                temp = f.readlines()
                # format i x y w h
                self.boxes = {int(line.split()[0]):list(map(lambda x : int(float(x)), line.split()[1:])) for line in temp if int(line.split()[0]) in indexes} # remove the first element which is the image index, result in x y w h

        else: # source is the path to the json file containing the boxes, already in the right format
            with open(source,"r") as f:
                self.boxes = json.load(f)
            self.boxes = {int(k):v for k,v in self.boxes.items() if int(k) in indexes} # filter on the indexes


        assert len(self.labels) == len(self.boxes) == len(self.img_names) == len(self.indexes) , "Length of labels, boxes, img_names, indexes are not equal"


    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, index):

        if self.split == "train" and not self.already_loaded:

            self.dino_transform = DiNORandomResizedCrop(size = (self.size,self.size),
                                                        dino_type="dinov2_vits14",
                                                        antialias=True) if self.use_dino else transforms.Compose([])
            self.already_loaded = True

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
                
            elif not self.use_dino:
                img = transforms.RandomResizedCrop((self.size,self.size))(img)
            else: # use dino
                img = transforms.ToTensor()(img)
                _, H, W = img.shape
                h = H + self.patch_size - H % self.patch_size
                w = W + self.patch_size - W % self.patch_size
                img = transforms.Resize((h,w),antialias=True)(img)
                img = self.dino_transform(img) 
                img = transforms.ToPILImage()(img.cpu())

        else: # val
            img = transforms.Resize(self.size)(img)
            img = transforms.CenterCrop(self.size)(img)


        # Postprocessing

        img = self.post_crop_transforms(img).float()

        label = torch.tensor(label).long()

        return img, label # img is a tensor, label is an int
 
from custom_transform import get_non_geo_transforms

def load_cub_datasets(cfg, all_vanilla = False):
    assert cfg.source == "gt" or (".json" in cfg.source), "source must be either gt or a path to a json file containing the boxes"
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
        
    with open(cfg.dataset.split_file,"r") as f:
        temp = f.readlines()
        # format i 1 or 0
        split = {int(line.split()[0]) : int(line.split()[1]) for line in temp}

    train_indexes = [i for i in split.keys() if split[i] == 1] # get the indexes of the train images

    val_indexes = [i for i in split.keys() if split[i] == 0] # get the indexes of the val images

    remaining_train = int(len(split)*ratio) - len(train_indexes)

    # split is approx 50%, take imgs from val to train to have a 70/30 split

    train_indexes += val_indexes[:remaining_train]

    val_indexes = val_indexes[remaining_train:]

    print(f"Train : {len(train_indexes)}")
    print(f"Val : {len(val_indexes)}")
    print(f"Split : {len(train_indexes)/(len(train_indexes)+len(val_indexes))}")

    """list_indexes = list(img_names.keys()) # basically a range(0,11788) but ensures that all indexes are present

    rd.seed(42) # hardcoded seed to prevent mixing train and val when transfering from a pretrained model
    rd.shuffle(list_indexes)

    train_indexes = list_indexes[:int(len(list_indexes)*ratio)]
    val_indexes = list_indexes[int(len(list_indexes)*ratio):]"""

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
                       use_dino = cfg.use_dino,
                       size = cfg.img_size,
                       patch_size = cfg.patch_size,
                       THR = cfg.THR,
                       
                       pre_crop_transforms= pre_crop_transforms,
                       post_crop_transforms= post_crop_transforms,

                       source =  cfg.source,

                        )
    
    val = CUBDataset(split="val",
                     path_to_img=cfg.dataset.img_dir,
                     img_names = {i:img_names[i] for i in val_indexes},
                     indexes = val_indexes,
                     label_file=cfg.dataset.label_file,
                     box_file=cfg.dataset.box_file,

                     use_box = False, # no box in val
                     use_dino = False,
                     size = cfg.img_size,
                     patch_size = cfg.patch_size,
                     THR = 0,

                     pre_crop_transforms= transforms.Compose([]), # no color augmentations
                     post_crop_transforms= transforms.Compose([transforms.ToTensor(),normalize]), # classic post crop transforms

                     source =  cfg.source,
                    
                     )
    
    if cfg.deterministic:
        seed = cfg.seed
        torch.manual_seed(seed)

    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers ,#if not cfg.use_dino else 0, # dino is not compatible with multiprocessing
                              pin_memory=True,
                              multiprocessing_context="spawn",
                              persistent_workers=True )#if not cfg.use_dino else False) 
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=True)
    
    return train_loader, val_loader
                        
    
class BboxDataset(Dataset):
    """
    Dataset for loading crops corresponding to bounding boxes
    """

    def __init__(self,
                 split : str, # train or val

                 path_to_img, 
                 img_names, 
                 indexes, 
                 label_file, 
                 box_file,
                 segmenation_file,

                 size : int,
                 patch_size,
                 THR,
                 ) -> None:
        super().__init__()

        self.split = split
        print(f"Loading {split} dataset")
        
        self.path_to_img = path_to_img
        self.img_names = img_names
        self.indexes = indexes
        self.label_file = label_file
        self.box_file = box_file
        self.segmenation_file = segmenation_file

        self.size = size
        self.patch_size = patch_size
        self.THR = THR
        

        # GT
        with open(label_file,"r") as f:
            temp = f.readlines()
            # format i class_number
            self.labels = {int(line.split()[0]) : int(line.split()[1])-1 for line in temp if int(line.split()[0]) in indexes}

        with open(box_file,"r") as f:
            temp = f.readlines()
            # format i x y w h
            self.boxes = {int(line.split()[0]):list(map(lambda x : int(float(x)), line.split()[1:])) for line in temp if int(line.split()[0]) in indexes} # remove the first element which is the image index, result in x y w h

        assert len(self.labels) == len(self.boxes) == len(self.img_names) == len(self.indexes) , "Length of labels, boxes, img_names, indexes are not equal"
        
        self.reverse_img_names = {v:k for k,v in self.img_names.items()} # path : index
        # Segmentation
        
        with open(segmenation_file,"r") as f:
            segmentation_dict = json.load(f)

        if self.split == "val": # unpack
            
            unpacked = []
            # segmentation_dict keys are the paths to the images + a setup entry
            for key in segmentation_dict.keys():
                if key != "setup" and key in self.img_names.values(): # filter on the indexes
                    for mask_info in segmentation_dict[key]:
                        unpacked.append((self.reverse_img_names[key],
                                        mask_info)) # (index, mask_info)
            
            self.segmentation = unpacked # list of (index, mask_info)

        else: # do not unpack
            self.segmentation = []

            for img_name in self.img_names.values():
                self.segmentation.append((self.reverse_img_names[img_name],
                                          {"bbox" :   segmentation_dict[img_name][0]["crop_box"],
                                           "crop_box":segmentation_dict[img_name][0]["crop_box"]
                                           } # crop_box is the full image
                ))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    
        """
        Note : the mask info is a dict containing a bbox key
        """

    def __len__(self):
        return len(self.segmentation)
    
    def __getitem__(self, index):
        """
        Returns the index of the image ad the mask info
        """
        img_index, mask_info = self.segmentation[index]
        img_path = self.img_names[img_index]
        
        img = Image.open(os.path.join(self.path_to_img,img_path)).convert("RGB")

        label = self.labels[img_index]


        # Crop
        x,y,w,h = mask_info["bbox"]

        if self.split == "val":
            gt_bbox = self.boxes[img_index]
        else:
            gt_bbox = mask_info["crop_box"]

        
        if self.split == "val":
            img = LocalizedRandomResizedCrop(img, x,y,w,h, 
                                            size=self.size, 
                                            patch_size=self.patch_size, 
                                            THR=self.THR, 
                                            relative_upper_scale=1.0, ratio=(1.0,1.0))
        else: # center crop on train
            img = transforms.Compose([
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size)
            ])(img)

        img = self.transform(img)

        box = torch.tensor([x,y,w,h],dtype=torch.int32)
        gt_bbox = torch.tensor(gt_bbox,dtype=torch.int32)
    
        return img, img_index, label, box, gt_bbox

        
def load_box_dataset(cfg, segmenation_file):
    ratio = 0.7 # don't change this

    with open(cfg.dataset.img_file,"r") as f:
        temp = f.readlines()
        # format i path
        img_names = {int(line.split()[0]):line.split()[1] for line in temp} # index : path
        
    with open(cfg.dataset.split_file,"r") as f:
        temp = f.readlines()
        # format i 1 or 0
        split = {int(line.split()[0]) : int(line.split()[1]) for line in temp}

    train_indexes = [i for i in split.keys() if split[i] == 1] # get the indexes of the train images

    val_indexes = [i for i in split.keys() if split[i] == 0] # get the indexes of the val images


    remaining_train = int(len(split)*ratio) - len(train_indexes)

    # split is approx 50%, take imgs from val to train to have a 70/30 split

    train_indexes += val_indexes[:remaining_train]

    val_indexes = val_indexes[remaining_train:]

    print(f"Train : {len(train_indexes)}")
    print(f"Val : {len(val_indexes)}")
    print(f"Split : {len(train_indexes)/(len(train_indexes)+len(val_indexes))}")
    """list_indexes = list(img_names.keys()) # basically a range(0,11788) but ensures that all indexes are present

    rd.seed(42) # hardcoded seed to prevent mixing train and val when transfering from a pretrained model
    rd.shuffle(list_indexes)

    train_indexes = list_indexes[:int(len(list_indexes)*ratio)]
    val_indexes = list_indexes[int(len(list_indexes)*ratio):]"""


    train_dataset = BboxDataset(split="train",
                                path_to_img=cfg.dataset.img_dir,
                                img_names = {i:img_names[i] for i in train_indexes},
                                indexes = train_indexes,
                                label_file=cfg.dataset.label_file,
                                box_file=cfg.dataset.box_file,
                                segmenation_file=segmenation_file,
                                
                                size = cfg.img_size,
                                patch_size = cfg.patch_size,
                                THR = 1.0,
                                )
    
    val_dataset = BboxDataset(split="val",
                                path_to_img=cfg.dataset.img_dir,
                                img_names = {i:img_names[i] for i in val_indexes},
                                indexes = val_indexes,
                                label_file=cfg.dataset.label_file,
                                box_file=cfg.dataset.box_file,
                                segmenation_file=segmenation_file,
                                
                                size = cfg.img_size,
                                patch_size = cfg.patch_size,
                                THR = 1.0,
                                )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=True)
    
    return train_loader, val_loader

    


if __name__ == "__main__":
    from config import cfg
    train_loader, val_loader = load_box_dataset(cfg, cfg.dataset.segmentation_file)

    print("train", len(train_loader))
    print("val", len(val_loader))
    print("Success")










