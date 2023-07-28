import torchvision.transforms.functional as F

from typing import Union

import random

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

        # for long object, we need to make sure the crop is not too big

        area_img = min(image.size[0], image.size[1]) ** 2

        area_obj = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) ** 2


        effective_scale = random.uniform(
                min(scale[0], area_obj / area_img),
                min(scale[1], area_obj / area_img / threshold),
        )

        effective_ratio = random.uniform(ratio[0], ratio[1])

        area_crop = effective_scale * area_img

        if area_crop > area_obj:
            range_x = ((area_crop / effective_ratio) ** 0.5 - (bbox[2] - bbox[0])) / 2
            range_y = ((area_crop * effective_ratio) ** 0.5 - (bbox[3] - bbox[1])) / 2

        else: # area_crop <= area_obj
            range_x = (bbox[2] - bbox[0] - area_crop ** 0.5/effective_ratio) / 2 + ((1-threshold)*area_crop)**0.5
            range_y = (bbox[3] - bbox[1] - area_crop ** 0.5*effective_ratio) / 2 + ((1-threshold)*area_crop)**0.5
            

        
        box_center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        crop_center = [
                box_center[0] + random.uniform(-range_x, range_x),
                box_center[1] + random.uniform(-range_y, range_y),
        ]

        crop_size = (area_crop * effective_ratio) ** 0.5

        crop_bbox = [
                crop_center[0] - crop_size / 2,
                crop_center[1] - crop_size / 2,
                crop_center[0] + crop_size / 2,
                crop_center[1] + crop_size / 2,
        ]

        crop_bbox = [int(x) for x in crop_bbox]

        return F.resized_crop(image, *crop_bbox, size, antialias=True)





