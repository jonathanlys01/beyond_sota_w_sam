import torchvision.transforms.functional as F

from typing import Union

import random

def LocalizedRandomResizedCrop(
                image, 
                bbox,
                size: Union[int, tuple],
                alpha: float = 0.5,
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


        bbox = [bbox[0], 
                bbox[1], 
                bbox[0] + bbox[2], 
                bbox[1] + bbox[3]]

        bbox = list(bbox)

        effective_scale = random.uniform(*scale)
        effective_ratio = random.uniform(*ratio)

        side = min(*image.size)

        crop_side = side * effective_scale

        Wc, Hc = crop_side * effective_ratio, crop_side # c -> cropped

        Wo, Ho = bbox[2] - bbox[0], bbox[3] - bbox[1] # o -> object

        range_x = alpha * (Wc + Wo)/2
        range_y = alpha * (Hc + Ho)/2

        effective_x = random.uniform(-range_x, range_x)
        effective_y = random.uniform(-range_y, range_y)

        center_x = (bbox[0] + bbox[2])/2 + effective_x
        center_y = (bbox[1] + bbox[3])/2 + effective_y

        crop_bbox = [
            max(0, center_x - Wc/2),
            max(0, center_y - Hc/2),
            min(image.size[0], center_x + Wc/2),
            min(image.size[1], center_y + Hc/2),
        ]

        crop_bbox = [int(x) for x in crop_bbox]

       
        return F.resized_crop(image, *crop_bbox, (size,size), antialias=True)






