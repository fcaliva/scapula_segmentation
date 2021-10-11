# Scapula segmentation

This repository contains the base code for the project of scapula segmentation.

For training, first prepare config file .yaml in the cfg folder, then run:
```
python main.py --cfg path_to_.yaml --desc model_name
```
For inference with labels, first prepare config file .yaml in the cfg folder, then run:
```
python main_infer.py --cfg path_to_.yaml --desc model_name
```
For inference without labels, first prepare config file .yaml in the cfg folder, then run: 
```
python main_infer_nolabel.py --cfg path_to_.yaml --desc model_name
```
Note that inference without labels was not tested for this project.

