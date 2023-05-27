from segment_anything import sam_model_registry
from utils.predictor import SamPredictor_mod
from utils.mod_funcs import *
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt


# initial config
checkpoint = 'model/sam_vit_h_4b8939.pth'
device = 'cuda'
sam = sam_model_registry['vit_h'](
    checkpoint=checkpoint).to(device)  # ViT-Huge
sam.image_encoder.eval()  # ViT-H image encoder (heavyweight)
sam.prompt_encoder.eval()  # SAM prompt encoder
sam.mask_decoder.train()
optimizer = torch.optim.RAdam(sam.mask_decoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


# Load batched image and mask label tensor
img = loadimg('images/train/2010_006079.jpg')  # NHWC
mask_label = loadmask('images/train/2010_006079-person-hair.png')  # NHW

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
        prompt_point = torch.flip(prompt_point_indices[sampled_indices], (1,))
        prompt_points.append(prompt_point)
    prompt_points = torch.stack(prompt_points, dim=0).to(device)
    # foreground ones, background zeros
    point_labels = torch.ones(batch_size).reshape(-1, 1).to(device)
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        # (N,n,2) n=number of points in a single img
        points=(apply_coords(prompt_points, original_size, sam=sam), point_labels),
        boxes=None,
        masks=None)

# 3. Mask Decoder Forward
low_res_masks, iou_predictions = sam.mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=sam.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,)

masks = sam.postprocess_masks(low_res_masks, input_size, original_size)
# return masks, iou_predictions, low_res_masks


# cast to numpy
prompt_points = prompt_points.cpu().numpy()
masks = masks.detach().cpu().numpy()
mask_label = mask_label.cpu().numpy()

# binarize
return_logits = False
if not return_logits:
    masks = masks > sam.mask_threshold

# visualize
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    for j in range(2):
        ax[i, j].imshow(img[0])
        ax[i, j].plot(prompt_points[0, 0, 0],
                      prompt_points[0, 0, 1], marker='*', ms=12, mec='white', mfc='green')
alpha = 0.6
ax[0, 0].imshow(mask_label[0], alpha=alpha)
ax[0, 1].imshow(masks[0, 0], alpha=alpha)
ax[1, 0].imshow(masks[0, 1], alpha=alpha)
ax[1, 1].imshow(masks[0, 2], alpha=alpha)
fig.tight_layout()
fig.savefig('test_NATIVE.png', dpi=200)
