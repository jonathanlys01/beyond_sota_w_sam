from segment_anything import sam_model_registry, SamAutomaticMaskGenerator # TODO resolve imports later
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
from config import cfg
import json

def generate_masks(amg, list_imgs, root,info):
    
    # output folder
    
    out_path = os.path.join(os.path.dirname(root), "masks_0")
    while os.path.exists(out_path):
        out_path = out_path[:-1] + str(int(out_path[-1]) + 1)
    os.mkdir(out_path)
    print(f"Saving masks and info to {out_path}")
    
    error_names = []
    
    d = {} # will store img_name: other info

    for img_name in tqdm(list_imgs):
        #print(img_name)
        img = cv2.imread(os.path.join(root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            masks = amg.generate(img)
        except Exception as e:
            error_names.append(f"Error with {img_name}: {e}")
            continue
        
        L_mask = np.array([mask["segmentation"] for mask in masks])
        
        np.savez_compressed(os.path.join(out_path, 
                                         os.path.basename(img_name).split(".")[0]), # /path/to/masks/001.Black_footed_Albatross_0001_796111.npy
                L_mask)
        
        new = [{key: mask[key] for key in mask.keys() if key not in ["segmentation",
                                                                     "predicted_iou",
                                                                     "stability_score"
                                                                     ]} for mask in masks]

        d[img_name] = new
    
    d["setup"] = info
    with open(os.path.join(out_path, "masks_info.json"), "w") as f:
        json.dump(d, f)     
        
    with open(os.path.join(out_path, "errors.txt"), "w") as f:
        string = f'Got {len(error_names)} errors out of {len(list_imgs)} images\n' + "\n".join(error_names)
        f.write(string)


def get_list_images_recursive(path):
    list_images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                list_images.append(
                    os.path.relpath(os.path.join(root, file), path)
                )
    return list_images

    
if __name__ == "__main__":
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    root = cfg.dataset.img_dir

    list_images = get_list_images_recursive(root)[:10]
    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print(f"Loaded model {model_type} from {sam_checkpoint}")
    
    # First run with fine parameters
    amg = SamAutomaticMaskGenerator(sam, points_per_side = 40,)
    info = {"model_type": model_type,
            "points_per_side": 40,}
    generate_masks(amg, list_images, root, info)
    
    # Second run with rougher parameters
    amg_rough = SamAutomaticMaskGenerator(sam, points_per_side = 16, 
                                          stability_score_thresh = 0.92,
                                          pred_iou_thresh=0.86,
                                          )
    
    info = {"model_type": model_type,
            "points_per_side": 16,
            "stability_score_thresh": 0.92,
            "pred_iou_thresh": 0.86,
            }
    generate_masks(amg_rough, list_images, root, info)
    

