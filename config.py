from box import Box


config = {
    "wandb" : True,
    "use_box" : False,
    "log_interval" : 1,
    "num_epochs" : 200,

    "img_size" : 224, #224 ,
    "patch_size" : 14, # DINOv2


    "num_workers" : 4, # 4 experiment on paralel
    "batch_size" : 32,

    "model" : {
        "type" : "resnet50", #     "dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14", "resnet50"
        "n_classes" : 200,
        "ema_step" : 32
    },

    "opt": {
        "type" : "sgd", # "sgd", "adam", "adamw"
        "lr" : 1e-3, 
        "momentum" : 0.9,
        "weight_decay" : 1e-4,

        "scheduler" : "step", # "cosine", "step", 
        "step_size" : 50,
        "gamma" : 0.5,

    },

    "alpha" : 0.5, # distance between crop and object

    "other": {
        "label_smoothing": 0.1,
        "ema_decay": 0.999,
    },

}

import socket

name = socket.gethostname()

if name.startswith('sl-tp-br') :
    config["dataset"] = {
        "img_dir" : "/nasbrain/datasets/CUB_200_2011/images/",
        "box_file" : "/nasbrain/datasets/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/nasbrain/datasets/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/nasbrain/datasets/CUB_200_2011/images.txt",
    },
elif name.startswith("someone"):
     config["dataset"] = {
        "img_dir" : "/mnt/data/CUB_200_2011/images/",
        "box_file" : "/mnt/data/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/mnt/data/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/mnt/data/CUB_200_2011/images.txt",
    },
else:
     raise Exception("What is your machine bro?")

cfg = Box(config)