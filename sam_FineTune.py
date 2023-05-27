from segment_anything import sam_model_registry
from utils.mod_funcs import *
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing
from threading import Thread
from torchvision.transforms import CenterCrop
from functools import partial


class SamDataset(Dataset):
    def __init__(self, path):
        self.original_imgs = glob.glob(f'{path}/*.jpg')[:10]
        self.path = path
        self.images = []
        self.mask_labels = []
        self.resize = CenterCrop((500, 500))

        # Single process
        self.loadimgs(self.original_imgs)

    def loadimgs(self, original_imgs):
        count = 0
        for img in original_imgs:
            name_file = os.path.basename(img).split('.')[0]
            for mask in glob.glob(f"{self.path}/{name_file}*.png"):
                # filename + person + subcls: coarse only
                if len(os.path.basename(mask).split('-')) >= 3:
                    self.images.append(self.transform(loadimg(img)))
                    self.mask_labels.append(self.transform(loadmask(mask)))
                    count += 1
                    if count % 500 == 0:
                        print(f'PID {os.getpid()} loaded {count} images')
        self.images = torch.cat(self.images, dim=0)
        self.mask_labels = torch.cat(self.mask_labels, dim=0)

    def transform(self, image):
        if len(image.shape) == 4:  # NHWC
            image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
            image = self.resize(image)  # resize to 500x500
            image = image.permute(0, 2, 3, 1)  # NCHW->NHWC
        elif len(image.shape) == 3:  # NHW
            image = self.resize(image)  # resize to 500x500
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.mask_labels[index]


def forward_sam(img: torch.FloatTensor, mask_label: torch.FloatTensor, return_logits: bool = False, numpy: bool = False, multimask_output: bool = True) -> torch.FloatTensor:
    """
    Prompt inputs are generated from a single pixel from mask label.

    Args:
        img (torch.FloatTensor): RGB image of torch.float32 type tensor with shape (N,H,W,C)
        mask_label (torch.FloatTensor): Mask label of torch.float32 type tensor with shape (N,H,W). Prompt inputs are generated from this mask label.
        return_logits (bool, optional): If True, output masks are thresholded to binary values. Turn off when .backward() call.
        numpy(bool, optional): If true, predicted masks are converted to CPU NumPy arrays.
        multimask_output(bool, optional): If true, output masks are three masks with different resolutions. If false, output masks are single mask with the same resolution as input image (the first, coarse mask returned only).

    Returns:
        masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
        'iou_predictions': (torch.Tensor) The model's predictions of mask quality, in shape BxC.
        'low_res_logits': (torch.Tensor) Low resolution logits with shape BxCxHxW, where H=W=256. 
                        Can be passed as mask input to subsequent iterations of prediction.
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
        multimask_output=multimask_output,)  # if False,
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
    train_dataloader = SamDataset('images/train')
    val_dataloader = SamDataset('images/val')

    # Batch size more than 1 cause error (due to multi-prompt)
    # https://github.com/facebookresearch/segment-anything/issues/277
    train_dataloader = DataLoader(
        train_dataloader, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataloader, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    
    # Training Loop
    steps=0
    steps_max=16
    scores_train=[]
    scores_val=[]
    max_score_val=0
    for epoch in range(10):
        score_train=0
        score_val=0
        # training batch loop
        sam.mask_decoder.train()
        for idx,batch in enumerate(train_dataloader):
            img_label, mask_label=batch
            img_label=img_label.to(device)
            mask_label=mask_label.to(device)
            # forward
            masks, iou_predictions, low_res_masks = forward_sam(img_label, mask_label,return_logits=True, multimask_output=False) # take only coarse mask
            # compute loss and grad
            loss=loss_fn(masks[:,0,...],mask_label)/steps_max
            loss.backward()
            steps+=1
            # backprop acuumulations
            if steps==steps_max:
                steps=0
                optimizer.step()
                for p in sam.mask_decoder.parameters():
                    p.grad=None
            # evaluate scores
            mask_label_logits=mask_label.type(torch.bool)
            mask_pred_logits=masks>sam.mask_threshold
            score_train+=(mask_pred_logits==mask_label_logits).sum()/(np.prod(mask_label.shape)*len(train_dataloader))
            print(f"Batch {idx+1}/{len(train_dataloader)}")
        scores_train.append(score_train)
        print (f'Epoch {epoch+1} train score: {score_train}')
        
        #validation batch loop
        sam.mask_decoder.eval()
        with torch.no_grad():
            for idx,batch in enumerate(val_dataloader):
                img_label, mask_label=batch
                img_label=img_label.to(device)
                mask_label=mask_label.to(device)
                # forward
                masks, iou_predictions, low_res_masks = forward_sam(img_label, mask_label,return_logits=True, multimask_output=False) # take only coarse mask
                # evaluate scores
                mask_label_logits=mask_label.type(torch.bool)
                mask_pred_logits=masks>sam.mask_threshold
                score_val+=(mask_pred_logits==mask_label_logits).sum()/(np.prod(mask_label.shape)*len(val_dataloader))
                print(f"Batch {idx+1}/{len(val_dataloader)}")
        scores_val.append(score_val)
        print (f'Epoch {epoch+1} val score: {score_val}\n')
        
        # End of epoch
        if max_score_val<score_val:
            max_score_val=score_val
            sam.mask_decoder.to('cpu')
            best_decoder_param=deepcopy(sam.mask_decoder.state_dict())
            torch.save(best_decoder_param,'model/finetuned_decoder.pt')
            sam.mask_decoder.to(device)
            
    # End of training
    torch.save(best_decoder_param,'model/finetuned_decoder_final.pt')