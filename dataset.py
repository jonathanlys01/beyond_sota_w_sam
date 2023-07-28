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

from xml.etree import ElementTree as ET

def extract_class_name(label_path):
    # format: oxford pet dataset
    tree = ET.parse(label_path)
    root = tree.getroot()

    class_name = root.find("object").find("name").text

    return class_name


class CustomDataset(Dataset):

    def __init__(self,img_dir,mask_dir, label_dir, transforms=None, use_box = False, size = 224) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir)

        self.labels = []

        print("Loading labels")
        for name in tqdm(self.img_names):
            name_label = name.split(".")[0] + ".xml"
            if name_label in os.list_dir(label_dir):
                self.labels.append(os.path.join(label_dir,name_label))
        

        assert len(self.img_names) == len(self.mask_names) == len(self.labels) , "Number of images and masks are not equal"

        
        
        if transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])

            self.transforms = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms



        self.use_box = use_box

        self.size = size




        

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        
        img = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.use_box:

            return self.transforms(img)
        
        else:

            mask = cv2.imread(os.path.join(self.mask_dir, self.mask_names[index]), 0)

            # get bounding box

            bbox = cv2.boundingRect(mask)

            # localized random resized crop

            img = LocalizedRandomResizedCrop(img, bbox, size = self.size)

            # apply transforms

            return self.transforms(img)
        


def load_datasets(cfg,img_size):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])

    train = CustomDataset(img_dir=cfg.dataset.train.root_dir,
                          mask_dir=cfg.dataset.train.mask_dir,
                          transforms=transforms.Compose([
                              transforms.RandomResizedCrop(img_size,antialias=True), # remove when using localized random resized crop
                              transforms.RandomHorizontalFlip(),
                              transforms.TrivialAugmentWide(),
                              transforms.ToTensor(),
                              normalize,
                              transforms.RandomErasing(0.1),
                              ]),
                          size=img_size,
                          use_box = cfg.use_box,
                        )
    
    val = CustomDataset(img_dir=cfg.dataset.val.root_dir,
                        mask_dir=cfg.dataset.val.mask_dir,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop(int(img_size*0.965)),
                            ]),
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
                        
    









