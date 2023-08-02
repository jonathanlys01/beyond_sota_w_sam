from box import Box


config = {
    "wandb" : True,
    "use_box" : True,
    "log_interval" : 1,
    "num_epochs" : 100,

    "img_size" : 224 ,
    "patch_size" : 14, # DINOv2


    "num_workers" : 8,
    "batch_size" : 64,

    "dataset" : {
        "img_dir" : "/mnt/data/CUB_200_2011/images/",
        "box_file" : "/mnt/data/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/mnt/data/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/mnt/data/CUB_200_2011/images.txt",
    },

    "model" : {
        "type" : "dinov2_vitl14", #     "dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14"
        #"MLP_dim" : 200,
        "n_classes" : 200
    },

    "opt": {
        "type" : "sgd", # "sgd", "adam", "adamw"
        "lr" : 5e-1, # 0.5 on mlp, 5e-3 on backbone
        "momentum" : 0.9,
        "weight_decay" : 2e-5,
    },

    "alpha" : 0.3, # distance between crop and object

    "other": {
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "ema_decay": 0.99,
    },

}

cfg = Box(config)