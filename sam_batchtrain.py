from segment_anything import sam_model_registry
from predictor import SamPredictor_mod
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import matplotlib.pyplot as plt


def loadimg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.FloatTensor(img).unsqueeze(0)  # NHWC


def loadmask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)/255
    return torch.FloatTensor(mask).unsqueeze(0)  # NHW


# initial config
checkpoint = 'model/sam_vit_h_4b8939.pth'
device = 'cuda'
sam = sam_model_registry['vit_h'](
    checkpoint=checkpoint).to(device)  # ViT-Huge
sam.image_encoder.eval()  # ViT-H image encoder (heavyweight)
sam.prompt_encoder.eval()  # SAM prompt encoder
sam.mask_decoder.train()
predictor = SamPredictor_mod(sam)
optimizer = torch.optim.RAdam(sam.mask_decoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


# load img and mask label
def apply_image(image: torch.tensor) -> torch.tensor:
    """
    Expects a torch tensor with shape NxHxWxC in uint8 format.
    """
    target_length = sam.image_encoder.img_size
    target_size = get_preprocess_shape(
        image.shape[1], image.shape[2], target_length)
    return resize(image.permute(0, 3, 1, 2), target_size)


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

# Load batched image and mask label tensor
img = torch.FloatTensor(loadimg('images/train/2008_000008.jpg'))  # NHWC
mask_label = torch.FloatTensor(
    loadmask('images/train/2008_000008-person-luarm.png'))  # NHW
input_image = apply_image(img).to(device)
input_image = sam.preprocess(input_image)
image_embedding = sam.image_encoder(input_image)


# Image Decode + Batchdim

# make prompt from mask label
rng = np.random.default_rng()
prompt_point_indices = np.argwhere(mask_label == 1)  # (H,W) =  (y,x) -> (x,y)
prompt_point = np.flip(rng.choice(prompt_point_indices, size=1, axis=0))
prompt_point_label = np.array([1])  # foreground 1, background 0


# forward
predictor.set_image(img)
masks, scores, logits = predictor.predict(
    point_coords=prompt_point, point_labels=prompt_point_label)

# visualize
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    for j in range(2):
        ax[i, j].imshow(img)
        ax[i, j].plot(prompt_point[0, 0],
                      prompt_point[0, 1], marker='*', ms=30, mec='white', mfc='green')
ax[0, 0].imshow(mask_label, alpha=0.5)
ax[0, 1].imshow(masks[0], alpha=0.5)
ax[1, 0].imshow(masks[1], alpha=0.5)
ax[1, 1].imshow(masks[2], alpha=0.5)
fig.tight_layout()
fig.savefig('testing.png', dpi=200)
