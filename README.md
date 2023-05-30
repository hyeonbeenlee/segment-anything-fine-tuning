# Fine tuning the Segment Anything Model (SAM)
## Summary
### This is a project repository for ***2023 spring semester Advanced Deep Learning***, Kyunghee University.  
### Team members: Hyeonbeen Lee, Keon-Oh Kim, Ji-Hyun Lee, Young-Hoon Cha  


![Screenshot 2023-05-31 060303](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/assets/78078652/084b7b4a-0be1-4592-9d7a-e502a8790bd7)  
![finetune](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/assets/78078652/f7552bbd-3f5f-44df-81f8-83fddd7f7e9f)

[sam_FineTune.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_FineTune.py) implements fine-tuning of the SAM mask decoder.  
[sam_forward.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/utils/sam_forward.py) implements ***batched SAM forward*** using ```torch``` and ```segment-anything.modeling.sam.Sam``` class.  
[sam_forward_SamPredictor.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/tmp/sam_forward_SamPredictor.py) implements unbatched SAM forward using ```segment-anything.SamPredictor``` class.  
[visualize.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/visualize.py) visualizes mask labels, SAM mask predictions, and fine-tuned SAM mask predictions.

A random single-pixel is sampled from the annotation label and prompted to the prompt encoder in ```utils.sam_forward.SamForward```

## Setup for the repository
### Packages
1. Install python packages.
Refer to [requirements.txt](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/requirements.txt) or if you're using ```conda``` environment,
```
conda create -n sam python=3.10
conda activate sam
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx numpy scipy 
```
and install [PyTorch.](https://pytorch.org/get-started/locally/)

### Datasets
2. Move to your working directory and clone this repo.
```
git clone https://github.com/hyeonbeenlee/segment-anything-fine-tuning.git
cd segment-anything-fine-tuning
```
3. Download PASCAL VOC 2010 train/val/test datasets.
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar
mkdir -p data/trainval
mkdir -p data/test
tar xvf VOCtrainval_03-May-2010.tar -C data/trainval
tar xvf VOC2010test.tar -C data/test
```


4. Download PASCAL-Part annotations.
```
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
mkdir -p data/annotations
tar xvzf trainval.tar.gz -C data/annotations  
```
5. Run data processing code [dataprocess.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/dataprocess.py)
```
python dataprocess.py
```

### Models
6. Download pretrained ViT-H base SAM model. 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir model
mv sam_vit_h_4b8939.pth model
```
Now you're good to go!

## Misc.
Coded based on https://github.com/facebookresearch/segment-anything with minimal changes.  
Thanks to @zuck
