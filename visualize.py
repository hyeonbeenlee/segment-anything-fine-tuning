import torch
import cv2
from sam_FineTune import forward_sam
from segment_anything import sam_model_registry
from utils.functions import loadimg
from utils.functions import loadmask
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.visual import *
import glob
import shutil
from random import sample


def plot_mask(img_path, mask_label_path):
    # train img
    img = loadimg(img_path)
    mask_label = loadmask(mask_label_path)

    # validation img
    # img = loadimg('images/val/2010_005991.jpg')
    # mask_label = loadmask('images/val/2010_005991-person-rlarm.png')

    with torch.no_grad():
        torch.manual_seed(0)
        mask, _, __, prompt = forward_sam(
            sam, img, mask_label, return_logits=True, multimask_output=False, return_prompt=True)
        torch.manual_seed(0)
        mask_tuned, _, __, prompt_ = forward_sam(
            sam_tuned, img, mask_label, return_logits=True, multimask_output=False, return_prompt=True)
    prompt = prompt.cpu().numpy()
    masks = [mask_label.unsqueeze(0), mask, mask_tuned]
    titles = ['Ground Truth', 'Before Fine-tuning', 'After Fine-tuning']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax[i].imshow(img.squeeze(0).type(torch.uint8).numpy())
        ax[i].imshow(masks[i].squeeze(0, 1).cpu().numpy()*255, alpha=0.5)
        ax[i].plot(prompt[..., 0], prompt[..., 1], marker='*',
                   mfc='red', ms=15, mec='white',)
        ax[i].set_title(titles[i])
        ax[i].grid(False)
    fig.suptitle('.'.join(os.path.basename(mask_label_path).split('.')[:-1]))
    fig.tight_layout()
    fig.savefig(
        f'{targets_path}_predictions/{os.path.basename(mask_label_path)}')
    plt.close('all')


def plot_log():
    loss_history = sam_tuned_log['loss_train']
    score_history = sam_tuned_log['scores_train']
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(loss_history)
    # ax[0].set_yscale('log')
    ax[0].set_xlabel('Batch updates')
    ax[0].set_ylabel('Loss')
    ax[1].plot(score_history)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Batch updates')
    ax[1].set_ylabel('Mask match rate')
    fig.tight_layout()
    fig.savefig('test_log.png')
    plt.close('all')


if __name__ == '__main__':
    # load original model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint).to(device)  # ViT-Huge

    # load fine-tuned decoder
    model_path = 'model/finetuned_decoder_epoch01_batch0003_score0.9085.pt'
    sam_tuned = deepcopy(sam)
    sam_tuned.mask_decoder.load_state_dict(torch.load(model_path))

    sam_tuned_log = torch.load(model_path+'log')

    plot_template()
    plot_log()
    # quit()
    targets_path = 'images/train'

    shutil.rmtree(targets_path+'_predictions')
    os.makedirs(targets_path+'_predictions')
    original_imgs = sample(glob.glob(f'{targets_path}/*.jpg'), k=100)
    for img in original_imgs:
        name = '.'.join(os.path.basename(img).split('.')[:-1])
        for mask_label in glob.glob(f'{targets_path}/{name}*.png'):
            name_mask = '.'.join(os.path.basename(mask_label).split('.')[:-1])
            name_mask = '-'.join(name_mask.split('-')[1:])
            if len(name_mask.split('-')) > 1:
                plot_mask(img, mask_label)
