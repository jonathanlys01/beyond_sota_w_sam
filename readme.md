# Experiments using SAM to crop images and enhance training for SOTA Resnet


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

