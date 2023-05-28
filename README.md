# Fine tuning the segment-anything model
Project for Advanced Deep Learning, Spring 2023  
[sam_forward_NATIVE.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_forward_NATIVE.py) implements ***batched SAM forward*** using ***torch*** and ***segment-anything.modeling.sam.Sam*** class.  
[sam_forward_SamPredictor.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_forward_SamPredictor.py) implements unbatched SAM forward using *segment-anything.SamPredictor* class provided by the Meta Research team.  
[sam_FineTune.py](https://github.com/hyeonbeenlee/segment-anything-fine-tuning/blob/master/sam_FineTune.py) implements fine-tuning SAM mask decoder.  

## Datasets
Download PASCAL VOC 2010 image dataset from here: [http://host.robots.ox.ac.uk:8080/eval/challenges/voc2010/  ](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/#devkit)   
tar xvzf FILENAME -C data/trainval  
tar xvzf FILENAME -C data/test  


Download PASCAL-Part annotations from here: http://roozbehm.info/pascal-parts/pascal-parts.html  
tar xvzf FILENAME -C data/annotations  

## Models
Download pretrained ViT-H base SAM model here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Misc.
Coded based on https://github.com/facebookresearch/segment-anything with minimal changes

The pretrained SAM mask decoder is fine tuned to PASCAL VOC 2010 person-part segmentation task.  
A random single-pixel is sampled from the annotation label and prompted to the prompt encoder.
