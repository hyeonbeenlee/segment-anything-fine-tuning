from xml.etree.ElementTree import parse
from scipy.io import loadmat
from copy import deepcopy
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt



def xml_parse():
    path = 'data/trainval/VOCdevkit/VOC2010/Annotations/2007_000027.xml'
    tree = parse(path)  # xml 파일을 트리 구조로 파싱
    root = tree.getroot()  # 트리의 시작점 지정

    keys = ['bndbox',]
    values_bndbox = ['xmin', 'xmax', 'ymin', 'ymax']
    for obj in root.iter('object'):
        for v in values_bndbox:
            # find(key).findtext(value)
            bndbox = int(obj.find('bndbox').findtext(v))
            print(f"{v}: {bndbox}")


def loadanno(path):
    train_list=open('data/trainval/VOCdevkit/VOC2010/ImageSets/Main/person_train.txt')
    val_list=open('data/trainval/VOCdevkit/VOC2010/ImageSets/Main/person_val.txt')
    trains=[]
    vals=[]
    for f in train_list.readlines():
        f=f.replace('\n','')
        if f.split()[1]=='1':
            trains.append(f.split(' ')[0])
    for f in val_list.readlines():
        f=f.replace('\n','')
        if f.split()[1]=='1':
            vals.append(f.split(' ')[0])
    
    
    file = loadmat(path)['anno']
    # file level
    name_file = file[0][0][0].item()
    data_file = file[0][0][1]
    if name_file in trains:dsettype='train'
    elif name_file in vals:dsettype='val'
    else:return

    # cls level
    for d in data_file[0]:
        if d[0]=='person':
            name_cls=d[0].item()
            idx_cls=d[1].item()
            mask_cls=d[2]
            data_cls=d[3]
    if 'name_cls' not in locals().keys():return

    # subcls level
    if data_cls.shape[0]:
        name_subcls = [data_cls[0][i][0].item() for i in range(
            len(data_cls[0]))]  # ['head', 'lear', 'leye', 'reye', ...]
        mask_subcls = [data_cls[0][i][1] for i in range(len(data_cls[0]))]
    
    # copy original image
    os.makedirs(f'images/{dsettype}', exist_ok=True)
    origin="data/trainval/VOCdevkit/VOC2010/JPEGImages"
    destination="images"
    shutil.copyfile(f"{origin}/{name_file}.jpg",f"{destination}/{dsettype}/{name_file}.jpg")
    
    
    # visualize cls/subcls and save
    masks_to_show = [mask_cls]
    masks_to_show = masks_to_show + \
        mask_subcls if 'mask_subcls' in locals().keys() else masks_to_show
    masks_keys = [name_cls]
    masks_keys = masks_keys + \
        [f"{name_cls}-{n}" for n in name_subcls] if 'name_subcls' in locals().keys() else masks_keys      
    
    for j, mask in enumerate(masks_to_show):
        name_image = f"{name_file}-{masks_keys[j]}"
        cv2.imwrite(f"{destination}/{dsettype}/{name_image}.png", mask*255)
        print(f"PID {os.getpid()} saved {destination}/{dsettype}/{name_image}.png")
    
def process():
    annotation_files = glob.glob('data/annotations/Annotations_Part/*.mat')
    import multiprocessing
    pool = multiprocessing.Pool(processes=os.cpu_count())
    pool.map_async(loadanno, annotation_files)
    pool.close()
    pool.join()


if __name__ == "__main__":
    # Sinlge processing
    # for m in glob.glob('data/annotations/Annotations_Part/*.mat'):
        # loadanno(m)
    
    # Multiprocessing
    process()