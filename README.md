# TAFT-SE on Deeplab V3+
This repository is an official implementation of TAFT-SE on Deeplab V3+ of following paper:
[Task-Adaptive Feature Transformer with Semantic Enrichment for Few-Shot Segmentation](https://arxiv.org/abs/2202.06498) on arXiv.
## Dependencies
* The codes are tested on Ubuntu 18.04 with Python 3.6 and Pytorch 1.10.0 

## Dataset Preparation

* Download VOC 2012 dataset and SBD dataset

* Download and unzip `Binary_map_aug.zip` from https://github.com/icoz69/CaNet (CVPR 2019 CANet code)

* Enter the path of VOC 2012, Binary_map_aug, and folder to create Pascal 5i in `create_pascal5i.py`

* Run `create_pascal5i.py`

* Enter the path of VOC 2012, SBD, created Pascal 5i, and folder to create auxiliary labels in `create_pascal5i_aux_label.py`

* Run `create_pascal5i_aux_label.py`

* Download MSCOCO 2014 dataset

* Enter the path of MSCOCO dataset and path to save labels of COCO dataset in `create_coco_labels.py`.

* Run ` create_coco_labels.py`

* Enter the path of MSCOCO dataset, created COCO labels and path to save COCO-20i dataset in `create_coco20i.py`.

* Run `create_coco20i.py`

* Enter the path of Pascal-5i dataset and Pascal-5i aux labels in ` train_TAFT_SE_Deeplab_pascal5i.py `, ` test_TAFT_SE_Deeplab_pascal5i.py `, ` test_TAFT_SE_Deeplab_pascal5i_mse.py ` .

* Enter the path of COCO-20i dataset and COCO-20i aux labels in ` train_TAFT_SE_Deeplab_coco20i.py `,` test_TAFT_SE_Deeplab_coco20i.py `.

## Running the code

```
# Training 

python train_TAFT_SE_Deeplab_pascal5i.py –gpu {GPU device number} –n_shot {number of shots} –split {split number} –n_queries {number of queries}

python train_TAFT_SE_Deeplab_coco20i.py –gpu {GPU device number} –n_shot {number of shots} –split {split number} –n_queries {number of queries}
```
* Before Test, Enter the path of trained model in `test_TAFT_SE_Deeplab_pascal5i.py ` or `test_TAFT_SE_Deeplab_coco20i.py `.

```
# Test 

python test_TAFT_SE_Deeplab_pascal5i.py –gpu {GPU device number} –n_shot {number of shots} –split {split number} –n_queries {number of queries}

python test_TAFT_SE_Deeplab_coco20i.py –gpu {GPU device number} –n_shot {number of shots} –split {split number} –n_queries {number of queries}
```

```
# Multi-Scale Input Test 

python test_TAFT_SE_Deeplab_pascal5i_mse.py –gpu {GPU device number} –n_shot {number of shots} –split {split number} –n_queries {number of queries}
```

## Pretrained Model Preparation

* Download pretrained model from Google Drive link [[pretrained_model]](https://drive.google.com/file/d/1DA_Rd1HFB1ioP7xgOESRrk1OIOZtT2yF/view?usp=sharing) 
