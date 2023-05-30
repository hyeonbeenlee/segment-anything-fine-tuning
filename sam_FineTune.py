from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
from utils.functions import *
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing as mp
from torchvision.transforms import CenterCrop
from utils.sam_loss import SamLoss
from utils.sam_dataset import SamDataset
from utils.sam_forward import SamForward
import random


def main():
    """
    Fine-tunes SAM mask decoder using PASCAL VOC 2010 dataset (additional person-part annotations).
    SAM model maps: (image, prompt) -> (mask)
    The model is prompted with a random single point from mask label.
    Binary accuracy of thresholded mask predictions is monitored, and the decoder model is saved when the highest validation accuracy is achieved.
    """
    global sam, device
    # Load SAM model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint).to(device)  # ViT-Huge

    # Initial config
    # todo: layerwise learning rate decay of 0.8 not properly applied
    # todo: drop-path with rate of 0.4
    # todo: decreasing lr with factor of 10 at epoch 60000, 86666...not considered
    sam.image_encoder.eval()  # ViT-H image encoder (Freeze)
    sam.prompt_encoder.eval()  # SAM prompt encoder (Freeze)
    sam.mask_decoder.train()  # Lightweight mask decoder (To be tuned)
    optimizer = torch.optim.AdamW([{'params': sam.mask_decoder.parameters(
    ), 'lr': 8e-4, 'betas': (0.9, 0.999), 'weight_decay': 0.1}])  # LR= SAM final training lr(8e-6)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = SamLoss()

    # Load dataset
    # fixme: data loading seems weird
    train_dataloader = SamDataset('images/train')

    # Batch size more than 1 cause error (due to multi-prompt)
    # https://github.com/facebookresearch/segment-anything/issues/277
    train_dataloader = DataLoader(
        train_dataloader, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)

    # Training Loop
    steps = 0
    steps_max = 256  # gradient accumulation steps
    scores_train = []
    loss_train = []
    score_train = 0
    batched_loss_train = 0
    batch_count = 0
    for epoch in range(100):
        # training batch loop
        sam.mask_decoder.train()
        steps = 0
        for idx, batch in enumerate(train_dataloader):
            img_label, mask_label = batch
            img_label = img_label.to(device)
            mask_label = mask_label.to(device)
            # forward
            masks, iou_predictions, low_res_masks = SamForward(sam,
                                                               img_label, mask_label)  # take only coarse mask
            # compute loss and grad
            loss = loss_fn(masks[:, 0, ...], mask_label, iou_predictions)
            loss /= steps_max
            loss.backward()
            batched_loss_train += loss.item()
            steps += 1
            # evaluate scores with logits
            with torch.no_grad():
                mask_label_logits = mask_label.type(torch.bool)
                mask_pred_logits = masks > sam.mask_threshold
                score_train += SamLoss().iou_logits(mask_pred_logits, mask_label_logits).item()/steps_max
            # update acuumulated grads
            if steps == steps_max or idx == len(train_dataloader)-1:
                print(
                    f"Epoch {epoch+1}, stepping at batch {idx+1}/{len(train_dataloader)}, mIoU score={score_train:.5f}, loss={batched_loss_train:.5f}")
                # record score log
                scores_train.append(score_train)
                loss_train.append(batched_loss_train)

                # backprop acuumulations
                optimizer.step()
                for p in sam.mask_decoder.parameters():
                    p.grad = None
                batch_count += 1

                # End of every update
                name = f"finetuned_decoder_epoch{epoch+1:02d}_batch{batch_count:04d}_score{score_train:.4f}"
                sam.mask_decoder.to('cpu')
                best_decoder_param = deepcopy(sam.mask_decoder.state_dict())
                sam.mask_decoder.to(device)
                torch.save(best_decoder_param, f'model/{name}.pt')

                log_dict = {"scores_train": scores_train,
                            "loss_train": loss_train}
                torch.save(log_dict, f'model/{name}.ptlog')

                # initialize
                steps = 0
                batched_loss_train = 0
                score_train = 0


if __name__ == '__main__':
    main()
