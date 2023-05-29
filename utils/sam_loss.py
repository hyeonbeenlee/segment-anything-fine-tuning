from .functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_focal = 20
        self.w_dice = 1
        self.w_focal = self.w_focal/(self.w_focal+self.w_dice)
        self.w_dice = self.w_dice/(self.w_focal+self.w_dice)

    def iou_logits(self, inputs, targets):
        eps = 1e-5
        intersection = torch.sum(inputs*targets, dim=(-2, -1))
        union = torch.sum(inputs, dim=(-2, -1)) + \
            torch.sum(targets, dim=(-2, -1))-intersection
        iou = torch.mean((intersection+eps)/(union+eps), dim=0)
        return iou

    def iou_loss(self, inputs, targets, iou_predictions):
        # inputs: NC
        # https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
        eps = 1e-5
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs*targets, dim=(-2, -1))
        union = torch.sum(inputs, dim=(-2, -1)) + \
            torch.sum(targets, dim=(-2, -1))-intersection
        iou_label = (intersection+eps)/(union+eps)
        iou_loss = torch.mean(torch.square(iou_label-iou_predictions))
        return iou_loss

    def dice_loss(self, inputs, targets, eps=1e-5):
        # inputs: NCHW
        # targets: NCHW
        # https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs*targets, dim=(-2, -1))
        union = torch.sum(inputs, dim=(-2, -1)) + \
            torch.sum(targets, dim=(-2, -1))-intersection
        # dice coefficient
        dice = torch.mean(2.0 * (intersection + eps) / (union + eps), dim=0)
        # dice loss
        dice_loss = 1.0 - dice
        return dice_loss

    def focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        # https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    def forward(self, mask_pred, mask_label, iou_predictions):
        # assume (C,H,W) images
        Lf = self.focal_loss(mask_pred, mask_label)
        Ld = self.dice_loss(mask_pred, mask_label)
        Li = self.iou_loss(mask_pred, mask_label, iou_predictions)
        return self.w_focal*Lf+self.w_dice*Ld+Li
