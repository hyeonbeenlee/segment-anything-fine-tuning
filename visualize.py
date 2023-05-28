import torch
import cv2
from sam_FineTune import forward_sam
from segment_anything import sam_model_registry
from utils.mod_funcs import loadimg
from utils.mod_funcs import loadmask
from copy import deepcopy
import matplotlib.pyplot as plt

def plot_mask():
    # train img
    img = loadimg('images/train/2008_000008.jpg')
    mask_label = loadmask('images/train/2008_000008-person-torso.png')
    
    # validation img
    # img = loadimg('images/val/2010_005991.jpg')
    # mask_label = loadmask('images/val/2010_005991-person-rlarm.png')

    with torch.no_grad():
        mask, _, __ = forward_sam(
            sam, img, mask_label, return_logits=True, multimask_output=False)
        mask_tuned, _, __ = forward_sam(
            sam_tuned, img, mask_label, return_logits=True, multimask_output=False)

    masks = [mask_label.unsqueeze(0), mask, mask_tuned]
    titles= ['Ground Truth', 'Before Fine-tuning', 'After Fine-tuning']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        # axes[i].imshow(img.squeeze(0))
        axes[i].imshow(masks[i].squeeze(0, 1).cpu().numpy(), alpha=0.5)
        axes[i].set_title(titles[i])
    fig.tight_layout()
    fig.savefig('test_finetuned.png')


if __name__ == '__main__':
    # load original model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint).to(device)  # ViT-Huge

    # load fine-tuned decoder
    sam_tuned = deepcopy(sam)
    sam_tuned.mask_decoder.load_state_dict(torch.load(
        'model/finetuned_decoder_epoch01_batch0013_score0.9616.pt'))
    
    plot_mask()