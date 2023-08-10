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
                patch_size: int = 14, # for ViT
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

        crop_bbox = [int(x) for x in crop_bbox]

        return F.resized_crop(image, *crop_bbox, (size,size), antialias=True)


def LocalizedRandomErase(image: Image.Image,
                         xo: int, yo: int, Wo: int, Ho: int,
                         THR: float = 0.5,
                         ):
    """
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

# lists of color/other related transformations (no geometric transformations)
# A transformation will be randomly selected from the list (some transformations are repeated with different parameters)


non_geo_transforms = [
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # randomly change brightness, contrast, saturation and hue
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # blur the image with a gaussian kernel with a random sigma
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ElasticTransform(alpha=1, sigma=50, alpha_affine=50), # apply an elastic transformation to the image
    T.ElasticTransform(alpha=1, sigma=25, alpha_affine=25),
    T.RandomPosterize(bits=4, p=1.0), # posterize the image
    T.RandomPosterize(bits=2, p=1.0),
    T.RandomSolarize(threshold=128, p = 1.0), # solarize the image
    T.RandomSolarize(threshold=192, p=1.0),
    T.RandomEqualize(p=1.0), # equalize the image
    T.RandomAutocontrast(p=1.0), #  autocontrast the image
    T.RandomHorizontalFlip(p=1.0), # flip the image horizontally
                    ]
