from box import Box


config = {
    "wandb" : False,
    "use_box" : True,
    "log_interval" : 1,
    "num_epochs" : 100,

    "img_size" : 224, #224 ,
    "patch_size" : 14, # DINOv2


    "num_workers" : 4, # 4 experiment on paralel
    "batch_size" : 32,

    "dataset" : {
        "img_dir" : "/mnt/data/CUB_200_2011/images/",
        "box_file" : "/mnt/data/CUB_200_2011/bounding_boxes.txt",
        "label_file" : "/mnt/data/CUB_200_2011/image_class_labels.txt",
        "img_file" : "/mnt/data/CUB_200_2011/images.txt",
    },

    "model" : {
        "type" : "dinov2_vitl14", #     "dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14"
        "MLP_dim" : 200,
        "n_classes" : 200
    },

    "opt": {
        "type" : "sgd", # "sgd", "adam", "adamw"
        "lr" : 1e-3, # 0.5 on head, 5e-3 on backbone
        "momentum" : 0.9,
        "weight_decay" : 1e-4,

        "scheduler" : "cosine", # "cosine", "step", 
        "step_size" : 30,
        "gamma" : 0.1,

    },

    "alpha" : 1, # distance between crop and object

    "other": {
        "label_smoothing": 0.1,
        "ema_decay": 0.99,
    },

}

cfg = Box(config)