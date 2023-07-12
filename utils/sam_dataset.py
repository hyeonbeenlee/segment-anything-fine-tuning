from .functions import *
import glob
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing as mp
from torchvision.transforms import CenterCrop
import random


class SamDataset(Dataset):
    def __init__(self, path):
        random.seed(0)
        self.original_imgs = sorted(glob.glob(f'{path}/*.jpg'))
        self.path = path
        self.images = []
        self.mask_labels = []
        self.resize = CenterCrop((500, 500))

        # Single process
        self.loadimgpaths(self.original_imgs)

        # Multiprocessing
        # self.loadimgpaths_mp(self.original_imgs)
        # self.catimgs()

    def loadimg(self, path):
        name_file = os.path.basename(path).split('.')[0]
        for mask in glob.glob(f"{self.path}/{name_file}*.png"):
            # filename + person + subcls: coarse only
            if len(os.path.basename(mask).split('-')) >= 3:
                self.images.append(self.transform(loadimg(path)))
                self.mask_labels.append(self.transform(loadmask(mask)))
                print(f'PID {os.getpid()} loaded {mask}')

    def loadimgpaths_mp(self, original_imgs):
        p = mp.Pool(8)
        p.map(self.loadimgpaths, original_imgs)
        p.close()
        p.join()

    def loadimgpaths(self, original_imgs):
        count = 0
        for img in original_imgs:
            name_file = os.path.basename(img).split('.')[0]
            for mask in sorted(glob.glob(f"{self.path}/{name_file}*.png")):
                exit_flag = False
                # filename + person + subcls: coarse only
                if len(os.path.basename(mask).split('-')) >= 3:
                    for key in ['hair', 'ear', 'eye', 'ebrow', 'mouth', 'nose']:  # DO NOT LOAD
                        if key in mask:
                            exit_flag = True
                            break
                        else:
                            continue
                    if exit_flag:
                        continue
                    else:
                        self.images.append(img)
                        self.mask_labels.append(mask)
                        count += 1
                        print(
                            f'PID {os.getpid()} checked {count} images: {mask}')
                    if count==10:
                        break
                if count==10:
                    break
            if count==10:
                break

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
        image=self.transform(loadimg(self.images[index])).squeeze(0)
        mask_label=self.transform(loadmask(self.mask_labels[index])).squeeze(0)
        return image, mask_label
