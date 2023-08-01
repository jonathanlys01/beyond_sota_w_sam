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

from random import shuffle


class CUBDataset(Dataset):

    def __init__(self, path_to_img, img_names, indexes, label_file, box_file, transforms = None, use_box = False, patch_size = 14, size = 224) -> None:
        super().__init__()

        self.path_to_img = path_to_img
        self.img_names = img_names
        self.indexes = indexes
        self.label_file = label_file
        self.box_file = box_file
        self.use_box = use_box
        self.patch_size = patch_size
        self.size = size

        if transforms is None:
            print("Using default transforms")
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            transforms = transforms.Compose([
                transforms.Resize((cfg.img_size,cfg.img_size)),
                transforms.ToTensor(),
                normalize
            ])
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
            img = LocalizedRandomResizedCrop(img, box, size = self.size, patch_size = self.patch_size, alpha=0.3)
            
        else:
            img = transforms.RandomResizedCrop((self.size,self.size))(img)

        img = self.transforms(img).float()


        return img, label # img is a tensor, label is a int
 

def load_cub_datasets(cfg, ratio = 0.7):


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])

    
    img_names = {}

    with open(cfg.dataset.img_file,"r") as f:
        temp = f.readlines()
        # format i path
        img_names = {int(line.split()[0]):line.split()[1] for line in temp} # index : path
        

    list_indexes = list(img_names.keys())

    shuffle(list_indexes)

    train_indexes = list_indexes[:int(len(list_indexes)*ratio)]
    val_indexes = list_indexes[int(len(list_indexes)*ratio):]

    train = CUBDataset(path_to_img=cfg.dataset.img_dir,
                        img_names = {i:img_names[i] for i in train_indexes},
                       indexes = train_indexes,
                       label_file=cfg.dataset.label_file,
                       box_file=cfg.dataset.box_file,
                          transforms=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.TrivialAugmentWide(),
                                transforms.ToTensor(),
                                normalize,
                                transforms.RandomErasing(0.1),
                                ]),
                          use_box = cfg.use_box,
                          size = cfg.img_size,
                          patch_size = cfg.patch_size
                            )
    
    val = CUBDataset(path_to_img=cfg.dataset.img_dir,
                     img_names = {i:img_names[i] for i in val_indexes},
                        indexes = val_indexes,
                        label_file=cfg.dataset.label_file,
                        box_file=cfg.dataset.box_file,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                            ]),
                        use_box = False,
                        size = cfg.img_size,
                        patch_size = cfg.patch_size
                        )

    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                                pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=True)
    
    return train_loader, val_loader
                        
    









