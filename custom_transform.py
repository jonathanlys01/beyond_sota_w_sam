import torchvision.transforms as transforms

import torchvision.transforms.functional as F

import torch.nn as nn

from typing import Optional, Union

def LocalizedRandomResizedCrop(
                image, 
                bbox,
                size: Union[int, tuple],
                threshold: float = 0.5,
                scale: tuple = (0.08, 1.0),
                ratio: tuple = (3. / 4., 4. / 3.),
                ):
    
        """
        Args :
            image (PIL Image): Image to be cropped.
            bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
            scale (tuple): Range of the random size of the cropped image compared to
                the original image size.
            ratio (tuple): Range of aspect ratio of the origin aspect ratio cropped
                image compared to the original image ratio. Default value is
                (3. / 4., 4. / 3.).

        Returns:
            PIL Image: Cropped image.
    
        """

        bbox = list(bbox)





        

    
        



