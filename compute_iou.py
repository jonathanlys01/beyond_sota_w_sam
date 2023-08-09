import PIL
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms as transforms
import random
import numpy as np

from config import cfg

import cv2
import os

from tqdm import tqdm

transform = RandomResizedCrop(224)


def LocalizedRandomResizedCrop(
                image, 
                xo, yo, Wo, Ho,
                size: tuple,
                THR: float,
                scale: tuple = (0.08, 1.0),
                ratio: tuple = (3. / 4., 4. / 3.),
                ):
    
        """
        Args :
            image (PIL Image): Image to be cropped.
            bbox (tuple): Bounding box coordinates (x, y, w, h)
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.

                should be a multiple of patch_size
            alpha (float): 0: centers are equal, 1 : centers are shifted
            scale (tuple): Range of the random size of the cropped image compared to
                the original image size.
            ratio (tuple): Range of aspect ratio of the origin aspect ratio cropped
                image compared to the original image ratio. Default value is
                (3. / 4., 4. / 3.).

        Returns:
            PIL Image: Cropped image.
    
        """


        area_image = image.size[0] * image.size[1]

        Ao = max(Wo, Ho)**2

        scale = (max(scale[0], (THR*Ao)/area_image),
                 scale[1])
        
        effective_scale = random.uniform(*scale)

        log_ratio = tuple(np.log(r) for r in ratio)

        effective_ratio = np.exp(random.uniform(*log_ratio))

        side = area_image**0.5  

        crop_side = side * effective_scale**0.5


        if effective_ratio > 1:
            Wc = effective_ratio * crop_side
            Hc = crop_side
        else:
            Wc = crop_side
            Hc = crop_side / effective_ratio



        Hc = min(Hc, float(image.size[1]))
        Wc = min(Wc, float(image.size[0]))

        alpha = 1 - THR**0.5

        range_x = alpha * (Wc + Wo)/2
        range_y = alpha * (Hc + Ho)/2

        effective_x = random.uniform(-range_x, range_x)
        effective_y = random.uniform(-range_y, range_y)

        center_x = xo + Wo/2 + effective_x
        center_y = yo + Ho/2 + effective_y
        crop_bbox = [

            max(0, center_y - Hc/2),
            max(0, center_x - Wc/2),
            
            Hc,
            Wc,
            
        ]

        if crop_bbox[0] + crop_bbox[2] > image.size[1]:
            crop_bbox[2] = image.size[1] - crop_bbox[0]

        if crop_bbox[1] + crop_bbox[3] > image.size[0]:
            crop_bbox[3] = image.size[0] - crop_bbox[1]


        crop_bbox = [int(x) for x in crop_bbox]


        return crop_bbox
        


def compute_niou(bbox,gt,):

    """
    Args:
        y, x, h, w

        warning: not the true iou, but intersection over area of the gt
    """

    x1, y1, w1, h1 = bbox
    x3, y3, w3, h3 = gt

    x2 = x1 + w1
    y2 = y1 + h1

    x4 = x3 + w3
    y4 = y3 + h3

    #  x1,y1,x2,y2,x3,y3,x4,y4

    inter_width = min(x2, x4) - max(x1, x3)
    inter_height = min(y2, y4) - max(y1, y3)

    if inter_width <= 0 or inter_height <= 0:
        return 0
    
    areaIntersection = inter_width * inter_height

    return areaIntersection / (w3 * h3)


with open(cfg.dataset.box_file) as f:
    temp = f.readlines()

    # format i x y w h

    boxes = {int(line.split()[0]):list(map(lambda x : int(float(x)), line.split()[1:])) for line in temp}

with open(cfg.dataset.img_file) as f:
    temp = f.readlines()

    # format i path

    images_name = {int(line.split()[0]):line.split()[1] for line in temp}





def get_miou(THR):
    miou_localized = []

    miou_random = []


    size = 224
    total = len(images_name)
    assert list(boxes.keys()) == list(images_name.keys())

    for i in range(total):
        name = images_name[i+1]
        box = boxes[i+1]

        img = PIL.Image.open(os.path.join(cfg.dataset.img_dir,name))

        #img = transforms.ToPILImage()(img)



        localised_box = LocalizedRandomResizedCrop(img, *box, size = (size,size), THR = THR)
        random_box = transform.get_params(img, transform.scale, transform.ratio)

        box = [box[1], box[0], box[3], box[2]] # x,y,w,h -> y,x,h,w

        iou_random = compute_niou(random_box,box)
        iou_localised = compute_niou(localised_box,box)


        miou_random.append(iou_random)
        miou_localized.append(iou_localised)

        print(f"{np.mean(miou_localized):.3f} | {np.mean(miou_random):.3f} | {str(i).zfill(len(str(total)))}/{total}", end="\r")

    return miou_localized, miou_random
     

if __name__ == "__main__":
    miou_localized, miou_random = get_miou(THR = 0.0955)
    print()
    print(f"miou_localized : {np.mean(miou_localized):.3f}")
    print(f"miou_random : {np.mean(miou_random):.3f}")
