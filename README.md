# Fine tuning the segment-anything model
## Summary
This is a project repository for Advanced Deep Learning, 2023 spring, Kyunghee University.
[sam_FineTune.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_FineTune.py) implements fine-tuning of the SAM mask decoder.  
[sam_forward_NATIVE.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_forward_NATIVE.py) implements ***batched SAM forward*** using ***torch*** and ***segment-anything.modeling.sam.Sam*** class.  
[sam_forward_SamPredictor.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_forward_SamPredictor.py) implements unbatched SAM forward using *segment-anything.SamPredictor* class.

## Setup for the repository
### Packages
Refer to [requirements.txt](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/requirements.txt) or if you're using conda environment,
```
conda create -n sam python=3.10
conda activate sam
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx numpy scipy 
```
and install [PyTorch.](https://pytorch.org/get-started/locally/)

### Datasets
1. Download PASCAL VOC 2010 image dataset from here: [http://host.robots.ox.ac.uk:8080/eval/challenges/voc2010/  ](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/#devkit)   
2. Then run:
```
mkdir -p data/trainval
mkdir -p data/test
tar xvzf <DOWNLOADED_FILEPATH> -C data/trainval
tar xvzf <DOWNLOADED_FILEPATH> -C data/test
```


3. Download PASCAL-Part annotations from here: http://roozbehm.info/pascal-parts/pascal-parts.html  
4. Then run:
```
mkdir -p data/annotations
tar xvzf <DOWNLOADED_FILEPATH> -C data/annotations  
```
5. Run [dataprocess.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/dataprocess.py)

### Models
1. Download pretrained ViT-H base SAM model here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
2. Then run:
```
mkdir model
mv <DOWNLOADED_FILEPATH> model/sam_vit_h_4b8939.pth
```
Now you're good to go!

## Misc.
Coded based on https://github.com/facebookresearch/segment-anything with minimal changes

The pretrained SAM mask decoder is fine tuned to PASCAL VOC 2010 person-part segmentation task.  
A random single-pixel is sampled from the annotation label and prompted to the prompt encoder.
