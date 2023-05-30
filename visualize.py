import torch
import cv2
from utils.sam_loss import SamLoss
from utils.sam_forward import SamForward
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
from torchvision.transforms import CenterCrop


def plot_mask(img_path, mask_label_path):
    img = loadimg(img_path)
    mask_label = loadmask(mask_label_path)

    with torch.no_grad():
        torch.manual_seed(0)
        mask, iou, __, prompt = SamForward(
            sam, img, mask_label, return_logits=True, multimask_output=False, return_prompt=True)
        torch.manual_seed(0)
        mask_tuned, iou_tuned, __, prompt_ = SamForward(
            sam_tuned, img, mask_label, return_logits=True, multimask_output=False, return_prompt=True)
    mask = mask.cpu()
    mask_tuned = mask_tuned.cpu()
    iou = iou.cpu()*100
    iou_tuned = iou_tuned.cpu()*100
    prompt = prompt.cpu().numpy()
    masks = [mask_label.unsqueeze(0), mask, mask_tuned]
    titles = ['Ground Truth', 'Before Fine-tuning', 'After Fine-tuning']
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    for i in range(3):
        ax[i].imshow(img.squeeze(0).type(torch.uint8).numpy(), alpha=0.9)
        ax[i].imshow(masks[i].squeeze(0, 1).numpy()*255,
                     alpha=0.7*(masks[i].numpy().squeeze((0, 1)) > 0),cmap='plasma',vmin=0,vmax=255)
        ax[i].plot(prompt[..., 0], prompt[..., 1], marker='*',
                   mfc='firebrick', ms=15, mec='white',mew=1)
        ax[i].set_title(titles[i])
        ax[i].grid(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    iou_label = SamLoss().iou_logits(mask, mask_label)*100
    iou_label_tuned = SamLoss().iou_logits(mask_tuned, mask_label)*100
    ax[1].set_xlabel(
        f"IoU = {iou_label.mean():.2f}\nPredicted IoU = {iou.mean():.2f}", fontsize=16)
    ax[2].set_xlabel(
        f"IoU = {iou_label_tuned.mean():.2f}\nPredicted IoU = {iou_tuned.mean():.2f}", fontsize=16)
    fig.suptitle('.'.join(os.path.basename(mask_label_path).split('.')[:-1]))
    fig.tight_layout()
    fig.savefig(
        f'{targets_path}_predictions/{os.path.basename(mask_label_path)}')
    print(
        f'Saved: {targets_path}_predictions/{os.path.basename(mask_label_path)}')
    plt.close('all')


def plot_log():
    loss_history = sam_tuned_log['loss_train']
    score_history = np.array(sam_tuned_log['scores_train'])*100
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(loss_history, color='k')
    # ax[0].set_yscale('log')
    ax[0].set_xlabel('Batch updates')
    ax[0].set_ylabel('Loss')
    ax[1].plot(score_history, color='k')
    ax[1].hlines(72.82, xmin=0, xmax=len(score_history)-1,
                 color='red', linestyle='--', label='SOTA')
    ax[1].set_ylim(0, 100)
    ax[1].set_xlabel('Batch updates')
    ax[1].set_ylabel('mIoU score')
    ax[1].legend(loc=2)
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig('training_result.png', dpi=200)
    plt.close('all')


def plot_predictions():
    shutil.rmtree(targets_path+'_predictions', ignore_errors=True)
    os.makedirs(targets_path+'_predictions', exist_ok=True)
    original_imgs = glob.glob(f'{targets_path}/*.jpg')
    for img in original_imgs:
        name = '.'.join(os.path.basename(img).split('.')[:-1])
        for mask_label in glob.glob(f'{targets_path}/{name}*.png'):
            exit_flag = False
            for key in ['hair', 'ear', 'eye', 'ebrow', 'mouth', 'nose']:
                if key in mask_label:
                    exit_flag = True
                    break
                else:
                    continue
            if exit_flag:
                continue
            else:
                name_mask = '.'.join(os.path.basename(
                    mask_label).split('.')[:-1])
                name_mask = '-'.join(name_mask.split('-')[1:])
                if len(name_mask.split('-')) > 1:
                    plot_mask(img, mask_label)


def compute_miou(sam):
    from utils.sam_loss import SamLoss
    metric = SamLoss()
    total_annotations_ = glob.glob(f'{targets_path}/*.png')
    total_annotations = total_annotations_.copy()
    for anno in total_annotations_:
        if len(anno.split('-')) < 3:
            total_annotations.remove(anno)
        elif len(anno.split('-')) == 3:
            for key in ['hair', 'ear', 'eye', 'ebrow', 'mouth', 'nose']:
                if key in anno:
                    total_annotations.remove(anno)
                    break
    print(f"Computing IoU scores on {targets_path}")

    scores = []
    scores_tuned = []
    count = 0
    for m in total_annotations:
        img = loadimg(m.split('-')[0]+'.jpg')
        mask_label = loadmask(m)
        # forward
        with torch.no_grad():
            mask, _, __ = SamForward(
                sam, img, mask_label, multimask_output=False)
        # logits
        mask_label = mask_label.type(torch.bool)
        mask = (mask > sam.mask_threshold).cpu()
        # evaluate
        score = metric.iou_logits(mask, mask_label)
        scores.append(score)
        count += 1
        print(
            f"{count}/{len(total_annotations)}: {score.item():.6f}")
    print()
    print(f"SAM: {torch.cat(scores).mean()}")


def plot_mask_labels():
    person = loadimg('images/val/2008_000261.jpg')[0].numpy().astype('uint8')
    masks = [
        loadmask('images/val/2008_000261-person-torso.png'),
        loadmask('images/val/2008_000261-person-ruleg.png'),
        loadmask('images/val/2008_000261-person-ruarm.png'),
        loadmask('images/val/2008_000261-person-rlarm.png'),
        loadmask('images/val/2008_000261-person-rhand.png'),
        loadmask('images/val/2008_000261-person-neck.png'),
        loadmask('images/val/2008_000261-person-luleg.png'),
        loadmask('images/val/2008_000261-person-lhand.png'),
        loadmask('images/val/2008_000261-person-head.png'),
    ]
    for i in range(len(masks)):
        masks[i] = masks[i][0].numpy().astype('uint8')
    color_indices = np.linspace(0, 255, len(
        masks), endpoint=True).astype('uint8')
    color_indices = np.random.randint(0, 255, size=len(masks))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(person)
    for i, p in enumerate(masks):
        ax.imshow(p*color_indices[i], alpha=0.6 *
                  (p > 0), cmap='rainbow', vmin=0, vmax=255)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('ppt_imgs/datadescription.png', dpi=200)


def plot_cropped_image():
    person = loadimg('images/val/2008_000261.jpg')[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(person.numpy().astype('uint8'))
    person = person.permute(2, 0, 1)
    person = CenterCrop((500, 500))(person)
    person = person.permute(1, 2, 0)
    person = person.numpy().astype('uint8')
    ax[1].imshow(person)
    for i in range(2):
        ax[i].grid(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.tight_layout()
    fig.savefig('ppt_imgs/cropped.png', dpi=200)


def plot_for_fun():
    """
    import matplotlib.pyplot as plt
    from utils.functions import loadimg
    img = loadimg('images/test.jpg')[0].numpy().astype('uint8')
    mark=dict(marker='*',ms=12,mec='white',mfc='red')
    plt.imshow(img)
    plt.grid()
    plt.plot(625,220,**mark) #head
    plt.plot(570,330,**mark) #neck
    plt.plot(650,600,**mark) # torso
    plt.plot(400,600,**mark) # ruarm
    plt.plot(500,790,**mark) # rlarm
    plt.plot(670,840,**mark) # rhand
    plt.plot(800,530,**mark) # luarm
    plt.plot(820,660,**mark) # llarm
    plt.plot(750,750,**mark) # lhand
    plt.plot(700,950,**mark) # ruleg
    plt.plot(850,770,**mark) # luleg
    plt.plot(1150,900,**mark) # llleg
    plt.plot(1400,980,**mark) # lfoot
    """
    img = loadimg('images/test.jpg')
    prompt_points = torch.FloatTensor([[625, 220], [570, 330], [650, 600], [400, 600], [500, 790], [
                                      670, 840], [800, 530], [820, 660], [750, 750], [700, 950], [850, 770], [1150, 900], [1400, 980]]).to(device)
    with torch.no_grad():
        masks = []
        masks_tuned = []
        for n in range(prompt_points.shape[0]):
            mask, iou, __ = SamForward(
                sam, img, return_logits=True, prompt_points=prompt_points[n].unsqueeze(0).unsqueeze(0))
            mask_tuned, iou, __ = SamForward(
                sam_tuned, img, return_logits=True, prompt_points=prompt_points[n].unsqueeze(0).unsqueeze(0))
            masks.append(mask.cpu().numpy().astype('uint8').squeeze((0, 1)))
            masks_tuned.append(
                mask_tuned.cpu().numpy().astype('uint8').squeeze((0, 1)))

    colors = np.linspace(0, 255, prompt_points.shape[0], endpoint=True)
    np.random.shuffle(colors)
    # colors = np.random.randint(0, 255, size=prompt_points.shape[0])
    mark = dict(marker='*', ms=12, mec='white', mfc='red')
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].imshow(img.squeeze(0).numpy().astype('uint8'))
    for n in range(prompt_points.shape[0]):
        for k in range(3):
            ax[k].plot(prompt_points[n].cpu().numpy()[0],
                    prompt_points[n].cpu().numpy()[1], **mark)
        ax[1].imshow(masks[n]*colors[n], alpha=0.3*(masks[n] > 0),
                     cmap='gnuplot', vmin=0, vmax=255)
        ax[2].imshow(masks_tuned[n]*colors[n], alpha=0.3 *
                     (masks[n] > 0), cmap='gnuplot', vmin=0, vmax=255)
    titles=['My Photo!', 'Before Fine-tuning', 'After Fine-tuning']
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(titles[i])
    fig.tight_layout()
    fig.savefig('ppt_imgs/for_fun.png', dpi=200)


if __name__ == '__main__':
    # load original model
    checkpoint = 'model/sam_vit_h_4b8939.pth'
    device = 'cuda'
    sam = sam_model_registry['vit_h'](
        checkpoint=checkpoint)  # ViT-Huge

    # load fine-tuned decoder
    model_path = 'model/finetuned_decoder_epoch04_batch0217_score0.5006.pt'
    sam_tuned = deepcopy(sam)
    sam_tuned.mask_decoder.load_state_dict(torch.load(model_path))
    sam_tuned_log = torch.load(model_path+'log')

    sam.to(device)
    sam_tuned.to(device)
    sam.eval()
    sam_tuned.eval()

    # validation data path
    targets_path = 'images/val'

    # plot
    os.makedirs('ppt_imgs', exist_ok=True)
    plot_template()
    plot_log()
    # plot_mask_labels()
    # plot_cropped_image()
    # plot_for_fun()
    
    plot_predictions()
    
    # compute_miou(sam)
    # compute_miou(sam_tuned)
