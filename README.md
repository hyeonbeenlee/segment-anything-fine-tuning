# Fine tuning the segment-anything model
Project for Advanced Deep Learning, Spring 2023

Download PASCAL VOC 2010 image dataset from here: http://host.robots.ox.ac.uk:8080/eval/challenges/voc2010/  
tar xvzf FILENAME -C data/trainval  
tar xvzf FILENAME -C data/test  


Download PASCAL-Part annotations from here: http://roozbehm.info/pascal-parts/pascal-parts.html  
tar xvzf FILENAME -C data/annotations  


Coded based on https://github.com/facebookresearch/segment-anything with minimal changes

The pretrained SAM mask decoder is fine tuned to PASCAL VOC 2010 person-part segmentation task.  
A random single-pixel is sampled from the annotation label and prompted to the prompt encoder.
