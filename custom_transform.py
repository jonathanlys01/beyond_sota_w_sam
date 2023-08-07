import torchvision.transforms.functional as F

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

        scale = (max(scale[0], THR*max(Wo,Ho)**2/area_image), scale[1])
        
        effective_scale = random.uniform(*scale)
        log_ratio = tuple(np.log(r) for r in ratio)

        effective_ratio = np.exp(random.uniform(*log_ratio))

        side = max(*image.size)

        crop_side = side * effective_scale**0.5

        Wc, Hc = crop_side * effective_ratio * max(*ratio), crop_side * max(*ratio)

        #alpha = 1 - 2 * ((THR*Ho*Wo)/(Ho+Hc)/(Wo+Wc))**0.5

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

        crop_bbox = [
                crop_bbox[0],
                crop_bbox[1],
                min(abs(crop_bbox[0]-image.size[1]), crop_bbox[2]),
                min(abs(crop_bbox[1]-image.size[0]), crop_bbox[3]),
        ]

        crop_bbox = [int(x) for x in crop_bbox]

        return F.resized_crop(image, *crop_bbox, (size,size), antialias=True)






