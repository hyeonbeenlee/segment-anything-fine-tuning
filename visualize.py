import torch
import cv2
from sam_FineTune import SamForward
from segment_anything import sam_model_registry
from utils.functions import loadimg
from utils.functions import loadmask
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.visual import *
import glob
import shutil
from random import sample
import numpy as np


def plot_mask(img_path, mask_label_path):
    img = loadimg(img_path)
    mask_label = loadmask(mask_label_path)

    with torch.no_grad():
        torch.manual_seed(0)
        mask, _, __, prompt = SamForward(
            sam, img, mask_label, return_logits=True, multimask_output=False, return_prompt=True)
        torch.manual_seed(0)
        mask_tuned, _, __, prompt_ = SamForward(
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
    score_history = np.array(sam_tuned_log['scores_train'])*100
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(loss_history, color='k')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Batch updates')
    ax[0].set_ylabel('Loss')
    ax[1].plot(score_history, color='k')
    ax[1].hlines(72.82, xmin=0, xmax=len(score_history)-1, color='red', linestyle='--', label='SOTA')
    ax[1].set_ylim(0, 100)
    ax[1].set_xlabel('Batch updates')
    ax[1].set_ylabel('mIoU score')
    ax[1].legend(loc=2)
    fig.tight_layout()
    fig.savefig('test_log.png',dpi=200)
    plt.close('all')


def plot_predictions():
    shutil.rmtree(targets_path+'_predictions', ignore_errors=True)
    os.makedirs(targets_path+'_predictions', exist_ok=True)
    original_imgs = glob.glob(f'{targets_path}/*.jpg')[:200]
    for img in original_imgs:
        name = '.'.join(os.path.basename(img).split('.')[:-1])
        for mask_label in glob.glob(f'{targets_path}/{name}*.png'):
            name_mask = '.'.join(os.path.basename(mask_label).split('.')[:-1])
            name_mask = '-'.join(name_mask.split('-')[1:])
            if len(name_mask.split('-')) > 1:
                plot_mask(img, mask_label)


def compute_miou(sam):
    from utils.sam_loss import SamLoss
    metric = SamLoss()
    total_annotations_ = glob.glob(f'{targets_path}/*.png')
    total_annotations=total_annotations_.copy()
    for anno in total_annotations_:
        if len(anno.split('-'))<3:
            total_annotations.remove(anno)
        elif len(anno.split('-'))==3:
            for key in ['hair', 'ear', 'eye', 'ebrow', 'mouth', 'nose']:
                if key in anno:
                    total_annotations.remove(anno)
                    break
    print(f"Computing IoU scores on {targets_path}")

    scores = []
    scores_tuned = []
    count=0
    for m in total_annotations:
        img = loadimg(m.split('-')[0]+'.jpg')
        mask_label = loadmask(m)
        # forward
        with torch.no_grad():
            mask, _, __ = SamForward(
                sam, img, mask_label, multimask_output=False)
        # logits
        mask_label=mask_label.type(torch.bool)
        mask=(mask > sam.mask_threshold).cpu()
        # evaluate
        score = metric.iou_logits(mask, mask_label)
        scores.append(score)
        count+=1
        print(
            f"{count}/{len(total_annotations)}: {score.item():.6f}")
    print()
    print(f"SAM: {torch.cat(scores).mean()}")


if __name__ == '__main__':
    # load original model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint)  # ViT-Huge

    # load fine-tuned decoder
    model_path = 'model/finetuned_decoder_epoch03_batch0166_score0.3910.pt'
    sam_tuned = deepcopy(sam)
    sam_tuned.mask_decoder.load_state_dict(torch.load(model_path))
    sam_tuned_log = torch.load(model_path+'log')

    
    sam.to(device)
    # sam_tuned.to(device)
    sam.eval()
    sam_tuned.eval()

    # validation data path
    targets_path = 'images/train'

    # plot
    # plot_template()
    # plot_log()
    # quit()
    # plot_predictions()
    compute_miou(sam)
    # compute_miou(sam_tuned)