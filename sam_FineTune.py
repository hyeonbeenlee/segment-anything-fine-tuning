from segment_anything import sam_model_registry
from utils.predictor import SamPredictor_mod
from utils.mod_funcs import *
from typing import Tuple
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing


class SamDataset(Dataset):
    def __init__(self, path):
        self.imgs = glob.glob(f'{path}/*.jpg')
        self.path=path
        self.dataset = []
        
        # self.loadimgs(self.imgs)
        
        pool = multiprocessing.Pool(processes=os.cpu_count())
        pool.map(self.loadimgs, self.imgs)
        pool.close()
        pool.join()
    
    def loadimgs(self,mainfiles):
        for img in mainfiles:
            name_file=os.path.basename(img).split('.')[0]
            for mask in glob.glob(f"{self.path}/{name_file}*.png"):
                if len(os.path.basename(mask).split('-'))>=3: # filename + person + subcls: coarse only
                    try:
                        self.dataset.append([loadimg(img), loadmask(mask)])
                    except:
                        continue

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, mask = self.dataset[index]
        return img, mask


def forward_sam(img: torch.FloatTensor, mask_label: torch.FloatTensor, return_logits: bool = False, numpy: bool = False, multimask_output:bool=True) -> torch.FloatTensor:
    """
    Prompt inputs are generated from a single pixel from mask label.

    Args:
        img (torch.FloatTensor): RGB image of torch.float32 type tensor with shape (N,H,W,C)
        
        mask_label (torch.FloatTensor): Mask label of torch.float32 type tensor with shape (N,H,W). Prompt inputs are generated from this mask label.
        
        return_logits (bool, optional): If True, output masks are thresholded to binary values. Turn off when .backward() call.
        
        numpy(bool, optional): If true, predicted masks are converted to CPU NumPy arrays.
        
        multimask_output(bool, optional): If true, output masks are three masks with different resolutions. If false, output masks are single mask with the same resolution as input image (the first, coarse mask returned only).
        
    Returns:
        torch.FloatTensor: Three masks of img with shape (N,3,H,W)
    """
    # SAM FORWARD
    with torch.no_grad():
        # 1. Image Encoder Forward
        batch_size = img.shape[0]
        original_size = img.shape[1:3]  # (H_org,W_org)
        input_image = apply_image(img, sam=sam).to(device)
        input_size = tuple(input_image.shape[-2:])  # (H_in,W_in)
        input_image = sam.preprocess(input_image)
        image_embeddings = sam.image_encoder(input_image)

        # 2. Create a random point prompt from mask_label and Prompt Encoder Forward
        prompt_points = []
        for i in range(batch_size):
            prompt_point_indices = torch.argwhere(mask_label[i] == 1)
            num_points = 1
            sampled_indices = torch.randint(
                prompt_point_indices.shape[0], size=(num_points,))
            # convert (H,W) =  (y,x) -> (x,y)
            prompt_point = torch.flip(
                prompt_point_indices[sampled_indices], (1,))
            prompt_points.append(prompt_point)
        prompt_points = torch.stack(prompt_points, dim=0).to(device)
        # foreground ones, background zeros
        point_labels = torch.ones(batch_size).reshape(-1, 1).to(device)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            # (N,n,2) n=number of points in a single img
            points=(apply_coords(prompt_points,
                    original_size, sam=sam), point_labels),
            boxes=None,
            masks=None)

    # 3. Mask Decoder Forward
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,) # if False, 
    masks = sam.postprocess_masks(low_res_masks, input_size, original_size)

    # binarize
    if not return_logits:
        masks = masks > sam.mask_threshold

    # cast to numpy
    if numpy:
        masks = masks.detach().cpu().numpy()

    return masks, iou_predictions, low_res_masks


if __name__ == '__main__':
    # Load SAM model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint).to(device)  # ViT-Huge

    # Initial config
    sam.image_encoder.eval()  # ViT-H image encoder (Freeze)
    sam.prompt_encoder.eval()  # SAM prompt encoder (Freeze)
    sam.mask_decoder.train()  # Lightweight mask decoder (To be tuned)
    optimizer = torch.optim.RAdam(sam.mask_decoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Load dataset
    # fixme: data loading seems weird
    train_dataloader = DataLoader(SamDataset(
        'images/train'), batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(SamDataset(
        'images/val'), batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    
    img,mask=next(iter(train_dataloader))
    pass