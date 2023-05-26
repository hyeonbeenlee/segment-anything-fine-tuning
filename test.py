import torch
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator


def segment(path,input_prompt=None):
    if 'heic' in path.lower():
        register_heif_opener()
        img=np.asarray(Image.open(path))
    else:
        img=cv2.imread(path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ##run by giving input prompts
    if input_prompt is not None:
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        masks, _, _ = predictor.predict(input_prompt)
    ## auto-gen masks
    else:
        mask_generator=SamAutomaticMaskGenerator(sam)
        masks=mask_generator.generate(img)
        
    ## visualize masks
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    axes[0].imshow(img)
    # from_func: showanns
    if len(masks) == 0:
        pass
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img_masked = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_masked[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img_masked[m] = color_mask
    axes[1].imshow(img_masked)
    
    axes[0].set_title('target')
    axes[1].set_title('masks')
    # save
    fig.tight_layout()
    fig.savefig('result.png',dpi=200)
    
    sam.prompt_encoder()
    
if __name__=='__main__':
    # initialize setup
    device='cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry["default"](checkpoint="model/sam_vit_h_4b8939.pth") # defaults to vit_h
    sam.to(device=device)
    
    segment('test.jpg')