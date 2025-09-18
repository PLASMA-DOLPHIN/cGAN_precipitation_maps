
import numpy as np
import cv2
from skimage import filters 
from utils import *
import matplotlib.pyplot as plt

def local_thres(patch):
   men = np.mean(patch)
   std = np.std(patch)
   #thres = men * (1+0.2*((std/128.0)-1))
   thres = filters.threshold_otsu(patch)
   mask = patch < thres 
   #mask = mask * 255
   return mask

def adaptive_binarize(image, patch_height, patch_width, overlap=0.3):


    overlap_width = int(patch_width * overlap)
    overlap_height = int(patch_height * overlap)
    
    height, width = image.shape
    
    binarized_image = np.zeros_like(image, dtype=np.uint8)
    
    for ys in range(0, height - patch_height, overlap_height):
        ye = ys + patch_height
        for xs in range(0, width - patch_width, overlap_width):
            xe = xs + patch_width
            patch = image[ys:ye, xs:xe]
            mask = local_thres(patch)
            binarized_image[ys:ye, xs:xe] = mask * 255

    for xs in range(0, width - patch_width, overlap_width):
        xe = xs + patch_width
        ye = height
        ys = ye - patch_height
        patch = image[ys:ye, xs:xe]
        mask = local_thres(patch)
        binarized_image[ys:ye, xs:xe] = mask * 255
        
    for ys in range(0, height - patch_height, overlap_height):
        ye = ys + patch_height
        xe = width
        xs = xe - patch_width
        patch = image[ys:ye, xs:xe]
        mask = local_thres(patch)
        binarized_image[ys:ye, xs:xe] = mask * 255
        
    ys = height - patch_height
    ye = height
    xs = width - patch_width
    xe = width
    patch = image[ys:ye, xs:xe]
    mask = local_thres(patch)
    binarized_image[ys:ye, xs:xe] = mask * 255
    
    return binarized_image