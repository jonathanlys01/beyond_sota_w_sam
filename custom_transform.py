import torchvision.transforms.functional as F

from PIL import Image

from typing import Union

import random
import numpy as np

def LocalizedRandomResizedCrop(
                image, 
                xo, yo, Wo, Ho,
                size: Union[int, tuple],
                THR: float = 0.5,
                scale: tuple = (0.08, 1.0),
                ratio: tuple = (3. / 4., 4. / 3.),
                patch_size: int = 14, # for ViT, TODO : adapt for ViT
                relative_upper_scale = None, # set to 1 when cropping bbox with non square aspect ratio (very stretched)
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

        if relative_upper_scale is None:
            scale = (max(scale[0], (THR*Ao)/area_image),
                    #min(scale[1], (1/(THR + 1e-8))*Ao/area_image))
                    scale[1])
        else:
             scale = (max(scale[0], (THR*Ao)/area_image),
                    min(scale[1], relative_upper_scale)*Ao/area_image)
        
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

        # prevent overflow
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

        crop_bbox = [abs(int(x)) for x in crop_bbox] # avoid crash with negative values

        return F.resized_crop(image, *crop_bbox, (size,size), antialias=True)


def LocalizedRandomErase(image: Image.Image,
                         xo: int, yo: int, Wo: int, Ho: int,
                         THR: float = 0.5,
                         ):
    """

    WIP
    
    Args : 

        image (PIL Image): Image to be erased.
        bbox (tuple): Bounding box coordinates (x, y, w, h)
        THR (float): threshold of the ratio of the erased area to the bbox area
    """
    H_i = image.size[1]
    W_i = image.size[0]

    d_up = H_i - (yo + Ho)
    d_down = yo

    d_left = xo
    d_right = W_i - (xo + Wo)

    d_v = d_up + d_down
    d_h = d_left + d_right

    u_v = random.uniform(0, 1)
    u_h = random.uniform(0, 1)

    if u_v > d_up / d_v:
        print('up')

    else:
        print('down')
    
    if u_h > d_left / d_h:
        print('left')
    else:
        print('right')


import torchvision.transforms as T

# list of color/other related transformations (no geometric transformations)
# A transformation will be randomly selected from the list (some transformations are repeated with different parameters)
# TODO: take transformations from trivial augmentation paper


def get_non_geo_transforms(p: float = 0.5):
        non_geo_transforms = [
        # randomly change brightness, contrast, saturation and hue
        T.ColorJitter(brightness=p*0.5, contrast=p*0.5, saturation=p*0.5, hue=p*0.1),

        # Posterize the image randomly with 4, 3, 2 bits
        T.RandomChoice(
            [T.RandomPosterize(bits=4, p=p), 
            T.RandomPosterize(bits=3, p=p),
            T.RandomPosterize(bits=2, p=p),]
        ),

        # Solarize the image randomly with thresholds 0.75, 0.9
        T.RandomChoice(
             [T.RandomSolarize(threshold=192, p = p),
              T.RandomSolarize(threshold=230, p = p),]
        ),

        # equalize the image
        T.RandomEqualize(p=p), 
        # autocontrast the image

        T.RandomAutocontrast(p=p),
                    ]
        return non_geo_transforms
