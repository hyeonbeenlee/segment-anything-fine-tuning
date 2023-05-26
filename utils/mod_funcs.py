from segment_anything import sam_model_registry
from utils.predictor import SamPredictor_mod
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import matplotlib.pyplot as plt
from copy import deepcopy


def loadimg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.FloatTensor(img).unsqueeze(0)  # NHWC


def loadmask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)/255
    return torch.FloatTensor(mask).unsqueeze(0)  # NHW


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    # batched torch.Tensor version of segment_anything.utils.transforms.ResizeLongestSide.get_preprocess_shape
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_image(image: torch.tensor, sam) -> torch.tensor:
    # batched torch.Tensor version of segment_anything.utils.transforms.ResizeLongestSide.apply_image
    """
    Expects a torch tensor with shape NxHxWxC in uint8 format.
    """
    target_length = sam.image_encoder.img_size
    target_size = get_preprocess_shape(
        image.shape[1], image.shape[2], target_length)
    return resize(image.permute(0, 3, 1, 2), target_size, antialias=True)

def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...], sam) -> np.ndarray:
    # batched torch.Tensor version of segment_anything.utils.transforms.ResizeLongestSide.apply_coords
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    target_length = sam.image_encoder.img_size
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length)
    coords = deepcopy(coords).type(torch.float32)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords
