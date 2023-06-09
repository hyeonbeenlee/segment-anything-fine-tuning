# Fine-tuning the Segment Anything Model (SAM)
## Summary
### This is a project repository for ***2023 spring semester Advanced Deep Learning***, Kyunghee University.  

We fine-tune the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) for human part segmentation task.  
The research is proposed mainly for two aspects: 
1. Evaluation of SAM on downstream tasks as the first foundation model for segmentation tasks.
2. Using the fine-tuned model, let collaborative robots recognize human parts and operate safely (future works).

![Screenshot 2023-05-31 060303](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/assets/78078652/084b7b4a-0be1-4592-9d7a-e502a8790bd7) 
![image](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/assets/78078652/7d2e1c28-a0df-4255-8d36-7678170263b1)  
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
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
curl -O http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar
mkdir -p data/trainval
mkdir -p data/test
tar xvf VOCtrainval_03-May-2010.tar -C data/trainval
tar xvf VOC2010test.tar -C data/test
```


4. Download PASCAL-Part annotations.
```
curl -O http://roozbehm.info/pascal-parts/trainval.tar.gz
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
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir model
mv sam_vit_h_4b8939.pth model
```
Now you're good to go!

## Existing Issues
### Errors
1. Batch size more than 1 cause error (due to multi-prompt)  
https://github.com/facebookresearch/segment-anything/issues/277d
Temporarily using unit-sized batch gradient accumulation
2. Multiprocessing image loading not properly working
3. <del>Custom dataset not implemented, which will load the entire training data to system memory at once (~300 GB).</del> Fixed at Jul 12th 2023.

### Not Implemented
1. Layerwise LR decay of 0.8
2. Drop-path with rate of 0.4
3. Decreasing LR with factor of 10 at iteration 60000, 86666...
4. The code trains **the first mask output** only, therefore the last two mask outputs of multi-mask outputs are wasted.



## Misc.
Coded based on https://github.com/facebookresearch/segment-anything with minimal changes.  
Thanks to @zuck
