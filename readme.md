# Experiments using DINO/SAM to crop images and enhance training for SOTA Resnet

## Introduction, motivation and goals

The goal of this project is to use knowledge from SAM and DINO to improve the training of a SOTA Resnet model. The baseline recipe can be found in the [baseline pruning repo](https://github.com/brain-bzh/baseline-pruning). It has been adapted by removing the cutmix and mixup augmentations (because the model perormed poorly with them), and adding color-wise augmentatons instead of geometry and random erasing that could affect the object of interest.

The idea is to use the self-attention maps from the DINO and SAM's masks to generate pseudo ground truth boxes for the images in the dataset. These boxes will be used to crop the images and enhance the training of the model.

Using those boxes in combination with the `LocalizedRandomResizedCrop` function will effectively improve the performance of a Resnet model trained with the baseline recipe by up to 0.6% accuracy.

This has been tested on the CUB dataset with different values of mean asymetric intersection over union (maIoU, defined as the intersection over the gt box). The result is a nearly affine relationship between the maIoU and the accuracy of the model. The accuracy after the first training is 84.97%. The 


<img src="assets/fig_1_acc_vs_maiou.png" alt="preliminary results" width="70%" position="center"/>

For reference, the running accuracy after the second training with the RandomResizedCrop is 85.01% on average with a standard deviation of 0.076% (over 9 runs).


Note that all the subsequent experiments are done on the CUB dataset with the Resnet50 model as a backbone.
## The `LocalizedRandomResizedCrop` function

The `LocalizedRandomResizedCrop` function takes a maiou threshold as a hyperparameter and generates a crop centered on the pseudo gt box with an aiou greater than the threshold. The crop is then resized to the input size of the model.

This works by generating a random shape for the crop (size and aspect ratio). The lower bound of the scale is constrained by the size of the pseudo gt box as the area of the crop has to be greater than 
the threshold times the area of the pseudo gt box. 
$$
scale_{eff}\sim U(\max(scale_{min},thr \frac{A_O}{A_I}),scale_{max})
$$
where $scale_{eff}$ is the effective scale of the crop, $scale_{min}$ and $scale_{max}$ are the lower and upper bounds of the scale, $A_O$ and $A_I$ are the areas of the pseudo gt box and the image.

Then, translation ranges are computed on both axes to move the crop around the pseudo gt box. 
The formula to compute the translation ranges is the following:
$$
\begin{cases}
      x_r = \alpha \frac{W_C+W_O}{2}\\
y_r = \alpha \frac{H_C+H_O}{2}
\end{cases}
$$
where $W_C$ and $H_C$ are the width and height of the crop, $W_O$ and $H_O$ are the width and height of the pseudo gt box and and $\alpha = 1 - \sqrt{thr}$.

Note that the parameter that is controlled is $thr$ and not the maIoU. It is possible to approximate the maIoU for a given $thr$ by computing the mean of the aiou for the crops generated with this $thr$.
<img src="assets/regression_maiou_thr.png" alt="regression" width="70%" position="center" />

The following approximation is not perfect but it is good enough to be used as a proxy for the maIoU.

$$
2(maIOU-0.5) =  thr^{0.4}
$$

## Using boxes

### SAM

Run generate_mask.py to generate masks for the images in the dataset. 

Masks format: 
- name: {image_name}.pt
- shape: (N_masks, H,W) where N_masks is the number of masks for the image, (H,W) is the shape of the og image
- device = cpu

The script will create 
- a folder masks_{i} (i is the number of the run)
- an error file called errors.txt
- an info file called masks_info.json
- all the masks 

The `generate_bbox.py` script will generate use the boxes from the masks and needs a pretrained model to run. The pseudo-gt boxes is the one with the maxiumum logits value for the gt class.

Computed IOU for the masks' bounding boxes and the ground truth bounding boxes is about 0.5

### DINO

It's possible to generate the pseudo gt boxes for the images by first thresholding the self-attention maps and the computing the following values:
```
center_y, center_x = np.mean(np.argwhere(threshed),axis=0)

h = np.sum(threshed.max(axis=1),axis=0)
w = np.sum(threshed.max(axis=0),axis=0)
```

Computed IOU for the pseudo gt boxes and the ground truth bounding boxes is about 0.6

## Using self-attention maps as 2D density maps

The implementation in the `dino_transform.py` is adapated from the `LocalizedRandomResizedCrop` from torchvision transforms, but generates the center of the crop using the self-attention maps as density maps.

These maps are extracted from the self-attention maps of the last layer of the transformer and cleaned using the following steps:
- Average over the heads
- Outlier removal (remove the max and replace it by the mean)**
- Apply a gaussian filter (Convolution with a gaussian kernel filled with ones)
- Aplly a min max normalization (optional)
- Apply a softmax

** It was observed that the max value of the self-attention maps was an outlier and was removed to avoid the crop to be centered on the same point for all the images.

The center of the crop is then sampled from the density map as discrete probability distribution. Thus, the more attention a point gets, the more likely it is to be sampled as the center of the crop. In general, the crop will be centered on the object of interest.

*This could be extended to the training of a transformer model in a self-distillation fashion, where we use the trained model to generate the density maps on the images where it performs well and train itself with the crops generated from those maps. This would allow the model to focus on the objects of interest and ignore the background, renforcing the attention on the objects of interest.*

## Results

### IOU for the boxes

| Model used | IOU | Total Time |
| --- | --- | --- |
| SAM | 50.12 | 7 hrs + 20 min* |  
| DINO | 61.57 | <15 min |

\* 7 hrs for the masks and 20 min for the boxes. 

SAM is much slower because
- it needs to generate the masks first, adding a very time-consuming step to the pipeline
- it really works better with the Huge Transformer model, while DINO already works well with the small one.
- the Automatic Mask Generator is quite slow, as it calls the mask decoder for each point in the prompt grid.

### Accuracy using the boxes

Then, the model is trained with the `LocalizedRandomResizedCrop` function using the pseudo gt boxes generated with DINO and a thr of 1. There are 2 configs for the box inference: using the vanilla DINO model and finetuning only the CLS token. The results are the following: **85.6%** with the vanilla model and **85.8%** with the finetuned model. This configuration matches the results obtained with the ground truth boxes, meaning that the pseudo gt boxes are good enough to be used for training.

To ensure that the results are not biased by the fact that CUB is part of the training set of the DINOv2 model, the train/val split of the dataset is consisent with the one used by DINOv2.
Also, the only information used by the Resnet model is spatial, not semantic.

### Accuracy using the self-attention maps as density maps

First, note that the GPU memory usage is much higher when using the self-attention maps as density maps. The maps are computed and not stored. It also slows down the dataloader as the maps are computed in the `__getitem__` method, making the loader process the whole dataset in twice the time it would take without the maps (8 min 36s vs 4 min 7s)
