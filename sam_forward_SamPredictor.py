from segment_anything import sam_model_registry
from utils.modified_sampredictor import SamPredictor_mod
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def loadimg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def loadmask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)/255
    return mask

# load img and mask label
img = loadimg('images/train/2010_006079.jpg')  # NHWC
mask_label = loadmask('images/train/2010_006079-person-hair.png')  # NHW

# make prompt from mask label
rng = np.random.default_rng()
prompt_point_indices = np.argwhere(mask_label == 1)
# (H,W) =  (y,x) -> (x,y)
prompt_point = np.flip(rng.choice(prompt_point_indices, size=1, axis=0))
prompt_point_label = np.array([1])  # foreground 1, background 0

# initial config
checkpoint = 'model/sam_vit_h_4b8939.pth'
device = 'cuda'
sam = sam_model_registry['vit_h'](
    checkpoint=checkpoint).to(device)  # ViT-Huge
sam.image_encoder.eval()  # ViT-H image encoder (heavyweight)
sam.prompt_encoder.eval()  # SAM prompt encoder
sam.mask_decoder.train()
predictor = SamPredictor_mod(sam)
# optimizer = torch.optim.RAdam(sam.mask_decoder.parameters(), lr=1e-4)
# loss_fn = torch.nn.MSELoss()

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
                      prompt_point[0, 1], marker='*', ms=12, mec='white',mfc='green')
alpha=0.5
ax[0, 0].imshow(mask_label, alpha=alpha)
ax[0, 1].imshow(masks[0], alpha=alpha)
ax[1, 0].imshow(masks[1], alpha=alpha)
ax[1, 1].imshow(masks[2], alpha=alpha)
fig.tight_layout()
fig.savefig('test_SamPredictor.png', dpi=200)