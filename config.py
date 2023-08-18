from box import Box


config = {
     
    "wandb" : True,
    "sweeprun" : True,

    "sweep" : {
        "resume" : True,
        "count" : None, # None for infinite
        "id" : "ffts7oug", #"0nq7lnoy", #"0fm2rs0t",
        
    },
     
    "save" : False,
    
    "augment": True, # use non geometric augmentations
    "augment_p": 1.0, # probability of using non geometric augmentations
    "use_augment_mix": False, # use augmix

    "use_box" : True,
    "log_interval" : 1,
    "num_epochs" : 200,

    "img_size" : 224, #224,
    "patch_size" : 14, # DINOv2


    "num_workers" : 4, # 4 experiment on paralel
    "batch_size" : 32,

    "deterministic" : False, # use deterministic training
    "seed" : 42,

    "model" : {
        "type" : "resnet50", #     "dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14", "resnet50"
        "n_classes" : 200,
        "ema_step" : 32,
        "freeze_backbone" : False,

        "resumed_model" : "/home/someone/stage_jonathan/beyond_sota_w_sam/models/r50-new-baseline_ac76.275_2023-08-11_09:26:58.pt", #"/home/someone/stage_jonathan/beyond_sota_w_sam/models/r50-baseline-500_ep_ac74.433_2023-08-09_02:07:41.pt", # None 
    },

    "opt": {
        "type" : "sgd", # "sgd", "adam", "adamw"

        "lr" : 2e-4, # 1e-3 from scratch
        "target_lr" : 1e-5,

        "momentum" : 0.9,
        "weight_decay" : 1e-4, # 5e-2 from baseline recipe, 1e-4 from repo

        "scheduler" : "step", # "cosine", "step", 
        "step_size" : 10,   
        "gamma" : 1, # true gamma will be automatically calculated

    },

    "THR": 1, # threshold for IoU, will change if a sweep is running

    "other": {
        "label_smoothing": 0.1,
        "ema_decay": 0.999, # 0.995 for 300ep, 0.99 for 100ep, when resuming, use 0.999
    },

}

import socket

name = socket.gethostname()

config['A_machine'] = name

# user (machine) specific config

if name.startswith('sl-tp-br') : # running on remote server
    config["dataset"] = {
        "img_dir" : "/nasbrain/datasets/CUB_200_2011/images",
        "box_file" : "/nasbrain/datasets/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/nasbrain/datasets/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/nasbrain/datasets/CUB_200_2011/images.txt",
    }
    config["batch_size"] = 32 # 32 for remote server (rip little 1060 gpu)
    config["num_workers"] = 4
    config["model"]["resumed_model"] = "/homes/j21lys/stage/beyond_sota_w_sam/models/r50-new-baseline_ac76.275_2023-08-11_09:26:58.pt" # "resnet50-cub-no_box_ac77.15_2023-08-06_21:08:14.pt" 
    # comment this line if you want to train from scratch

elif name.startswith("someone"): # running on local machine
    config["dataset"] = {
        "img_dir" : "/mnt/data/CUB_200_2011/images",
        "box_file" : "/mnt/data/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/mnt/data/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/mnt/data/CUB_200_2011/images.txt",

        "segmentation_file" : "/mnt/data/CUB_200_2011/masks_0/masks_info.json",
    }
    #config["num_workers"] = 8
else:
    raise Exception("Unrecognized machine")

cfg = Box(config)

import numpy as np

target_lr = cfg.opt.target_lr

start_lr = cfg.opt.lr

cfg.opt.gamma = np.exp(np.log(target_lr / start_lr) / (cfg.num_epochs / cfg.opt.step_size))