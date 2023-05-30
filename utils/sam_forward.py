from segment_anything.modeling.sam import Sam
from utils.functions import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def SamForward(sam: Sam, img: torch.FloatTensor, mask_label: torch.FloatTensor=None, return_logits: bool = False, numpy: bool = False, multimask_output: bool = False, device='cuda', return_prompt: bool = False, num_points: int = 1, prompt_points:bool=None) -> torch.FloatTensor:
    """
    Prompt inputs are generated from a single pixel from mask label.

    Args:
        img (torch.FloatTensor): RGB image of torch.float32 type tensor with shape (N,H,W,C)
        mask_label (torch.FloatTensor): Mask label of torch.float32 type tensor with shape (N,H,W). Prompt inputs are generated from this mask label.
        return_logits (bool, optional): If True, output masks are thresholded to binary values. Turn off when .backward() call.
        numpy(bool, optional): If true, predicted masks are converted to CPU NumPy arrays.
        multimask_output(bool, optional): If true, output masks are three masks with different resolutions. If false, output masks are single mask with the same resolution as input image (the first, coarse mask returned only).
        return_prompt(bool, optional): Returns randomly sampled prompt input if true
        num_points(int, optional): Number of randomly sampled prompt points from mask label. Defaults to 1.

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
        if mask_label is not None:
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
        else:
            assert prompt_points is not None, "Either mask_label or prompt_input must be provided."
        # foreground ones, background zeros
        point_labels = torch.ones((batch_size, num_points)).to(device)
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
