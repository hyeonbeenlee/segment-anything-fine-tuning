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
from torchvision.ops import focal_loss


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
        iou = torch.mean((intersection+eps)/(union+eps),dim=0)
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
        Ld = 1-self.dice_loss(mask_pred, mask_label)
        Li = self.iou_loss(mask_pred, mask_label, iou_predictions)
        return self.w_focal*Lf+self.w_dice*Ld+Li


class SamDataset(Dataset):
    def __init__(self, path):
        self.original_imgs = glob.glob(f'{path}/*.jpg')[:50]
        self.path = path
        self.images = []
        self.mask_labels = []
        self.resize = CenterCrop((500, 500))

        # Single process
        self.loadimgs(self.original_imgs)

        # todo: multiprocessing not working properly
        # Multiprocessing
        # self.loadimgs_mp(self.original_im
        self.catimgs()

    def loadimg(self, path):
        name_file = os.path.basename(path).split('.')[0]
        for mask in glob.glob(f"{self.path}/{name_file}*.png"):
            # filename + person + subcls: coarse only
            if len(os.path.basename(mask).split('-')) >= 3:
                self.images.append(self.transform(loadimg(path)))
                self.mask_labels.append(self.transform(loadmask(mask)))
                print(f'PID {os.getpid()} loaded {mask}')

    def loadimgs_mp(self, original_imgs):
        p = mp.Pool(8)
        p.map(self.loadimg, original_imgs)
        p.close()
        p.join()

    def catimgs(self):
        self.images = torch.cat(self.images, dim=0)
        self.mask_labels = torch.cat(self.mask_labels, dim=0)

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
                    print(f'PID {os.getpid()} loaded {count} images: {mask}')

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


def forward_sam(sam: Sam, img: torch.FloatTensor, mask_label: torch.FloatTensor, return_logits: bool = False, numpy: bool = False, multimask_output: bool = True, device='cuda', return_prompt: bool = False) -> torch.FloatTensor:
    """
    Prompt inputs are generated from a single pixel from mask label.

    Args:
        img (torch.FloatTensor): RGB image of torch.float32 type tensor with shape (N,H,W,C)
        mask_label (torch.FloatTensor): Mask label of torch.float32 type tensor with shape (N,H,W). Prompt inputs are generated from this mask label.
        return_logits (bool, optional): If True, output masks are thresholded to binary values. Turn off when .backward() call.
        numpy(bool, optional): If true, predicted masks are converted to CPU NumPy arrays.
        multimask_output(bool, optional): If true, output masks are three masks with different resolutions. If false, output masks are single mask with the same resolution as input image (the first, coarse mask returned only).
        return_prompt(bool, optional): Returns randomly sampled prompt input if true

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
    if return_logits:
        masks = masks > sam.mask_threshold

    # cast to numpy
    if numpy:
        masks = masks.detach().cpu().numpy()

    if return_prompt:
        return masks, iou_predictions, low_res_masks, prompt_points
    else:
        return masks, iou_predictions, low_res_masks


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
    ), 'lr': 8e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.2}])  # LR= SAM final training lr(8e-6)
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
    steps_max = 8  # gradient accumulation steps
    scores_train = []
    loss_train = []
    score_train = 0
    batched_loss_train = 0
    batch_count = 0
    for epoch in range(10):
        # training batch loop
        sam.mask_decoder.train()
        for idx, batch in enumerate(train_dataloader):
            img_label, mask_label = batch
            img_label = img_label.to(device)
            mask_label = mask_label.to(device)
            # forward
            masks, iou_predictions, low_res_masks = forward_sam(sam,
                                                                img_label, mask_label, return_logits=False, multimask_output=False)  # take only coarse mask
            # compute loss and grad
            loss = loss_fn(masks[:, 0, ...], mask_label, iou_predictions)
            loss /= steps_max
            loss.backward()
            batched_loss_train += loss.item()
            steps += 1
            # evaluate scores
            with torch.no_grad():
                mask_label_logits = mask_label.type(torch.bool)
                mask_pred_logits = masks > sam.mask_threshold
                score_train = SamLoss().iou_logits(mask_pred_logits, mask_label_logits).item()
            # acuumulated grads
            if steps == steps_max:
                print(
                    f"Epoch {epoch+1}, stepping at batch {idx+1}/{len(train_dataloader)},score={score_train:.5f} loss={batched_loss_train:.5f}")
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
