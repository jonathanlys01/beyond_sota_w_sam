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
                THR: float = 0.5,
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

        scale = (max(scale[0], (THR*Wo*Ho)/area_image), scale[1])
        
        effective_scale = random.uniform(*scale)
        log_ratio = tuple(np.log(r) for r in ratio)

        effective_ratio = np.exp(random.uniform(*log_ratio))

        side = (image.size[0]*image.size[1])**0.5  

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

        """if crop_bbox[0] + crop_bbox[2] > image.size[1]:
            crop_bbox[2] = image.size[1] - crop_bbox[0]

        if crop_bbox[1] + crop_bbox[3] > image.size[0]:
            crop_bbox[3] = image.size[0] - crop_bbox[1]"""


        crop_bbox = [int(x) for x in crop_bbox]

        """assert crop_bbox[0] >= 0 
        assert crop_bbox[1] >= 0 
        assert crop_bbox[0] + crop_bbox[2] <= image.size[1] 
        assert crop_bbox[1] + crop_bbox[3] <= image.size[0]"""

        
        return crop_bbox
        


def compute_iou(bbox,gt,):

    """
    Args:
        y, x, h, w

        warning: not the true iou, but intersection over area of the gt
    """

    y1, x1, h1, w1 = bbox
    y2, x2, h2, w2 = gt


    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1, x2+w2)
    yB = min(y1+h1, y2+h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = w1 * h1
    boxBArea = w2 * h2
    eps = 1e-5

    iou = interArea / (boxBArea + eps)

    return iou


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

        
        iou_random = compute_iou(random_box,box)
        iou_localised = compute_iou(localised_box,box)


        miou_random.append(iou_random)
        miou_localized.append(iou_localised)

        print(f"{np.mean(miou_localized):.3f} | {np.mean(miou_random):.3f} | {str(i).zfill(len(str(total)))}/{total}", end="\r")

    return miou_localized, miou_random
     

if __name__ == "__main__":
    miou_localized, miou_random = get_miou(THR = 1)
    print()
