from segment_anything import sam_model_registry, SamAutomaticMaskGenerator # TODO resolve imports later
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
from config import cfg
import json

def generate_masks(amg, list_imgs, root, sam_checkpoint, model_type, **kwargs):
    
    # output folder
    
    out_path = os.path.join(os.path.dirname(root), "masks_0")
    while os.path.exists(out_path):
        out_path = out_path[:-1] + str(int(out_path[-1]) + 1)
    os.mkdir(out_path)
    
    error_names = []
    
    d = {} # will store img_name: other info

    for img_name in tqdm(list_imgs):
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
        # list of dicts, each dict represents a mask, with keys:  
        # 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box' (not "mask")
        d[img_name] = new
    
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
    
    img_path = cfg.dataset.img_dir
    
    list_images = get_list_images_recursive(img_path)
    
    root = os.path.dirname(img_path) # get parent directory
    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    amg = SamAutomaticMaskGenerator(sam)
    
    # First run with default parameters (fine)
    generate_masks(amg, list_images, root, sam_checkpoint, model_type)
    
    """
    def __init__(model: Sam, 
    points_per_side: Optional[int]=32, 
    points_per_batch: int=64, 
    pred_iou_thresh: float=0.88, 
    stability_score_thresh: float=0.95, 
    stability_score_offset: float=1.0, 
    box_nms_thresh: float=0.7, 
    crop_n_layers: int=0, 
    crop_nms_thresh: float=0.7,
    crop_overlap_ratio: float=512 / 1500, 
    crop_n_points_downscale_factor: int=1, 
    point_grids: Optional[List[np.ndarray]]=None, 
    min_mask_region_area: int=0, 
    output_mode: str='binary_mask') -> None
    """
    
    amg_rough = SamAutomaticMaskGenerator(sam, points_per_side = 15, stability_score_thresh = 0.9)
    
    # Second run with rougher parameters
    generate_masks(amg_rough, list_images, root, sam_checkpoint, model_type)
    

