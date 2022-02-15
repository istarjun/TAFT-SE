from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

dataDir = 'PATH_TO_COCO2014_DATASET'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
label_dir = 'PATH_TO_VALIDATION_LABEL_DIRECTORY'

dataset_Dir = 'PATH_TO_CREATE_COCO20i_DATASET'

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

split_0 = []
split_1 = []
split_2 = []
split_3 = []

for i in range(20):
    split_0.append(nms[4*i])
    split_1.append(nms[4*i+1])
    split_2.append(nms[4*i+2])
    split_3.append(nms[4*i+3])

print('split 0 categories: \n{}\n'.format(' '.join(split_0)))
print('split 1 categories: \n{}\n'.format(' '.join(split_1)))
print('split 2 categories: \n{}\n'.format(' '.join(split_2)))
print('split 3 categories: \n{}\n'.format(' '.join(split_3)))

if not os.path.exists(dataset_Dir + '/0'):
    os.makedirs(dataset_Dir+'/0')
    for cl in split_0:
        os.makedirs(dataset_Dir+'/0/'+cl)
        os.makedirs(dataset_Dir+'/0/'+cl + '/train')
        os.makedirs(dataset_Dir+'/0/'+cl + '/train/groundtruth')
        os.makedirs(dataset_Dir+'/0/'+cl + '/test')
        os.makedirs(dataset_Dir + '/0/' + cl + '/test/groundtruth')

if not os.path.exists(dataset_Dir + '/1'):
    os.makedirs(dataset_Dir+'/1')
    for cl in split_1:
        os.makedirs(dataset_Dir+'/1/'+cl)
        os.makedirs(dataset_Dir+'/1/'+cl + '/train')
        os.makedirs(dataset_Dir+'/1/'+cl + '/train/groundtruth')
        os.makedirs(dataset_Dir+'/1/'+cl + '/test')
        os.makedirs(dataset_Dir + '/1/' + cl + '/test/groundtruth')

if not os.path.exists(dataset_Dir + '/2'):
    os.makedirs(dataset_Dir+'/2')
    for cl in split_2:
        os.makedirs(dataset_Dir+'/2/'+cl)
        os.makedirs(dataset_Dir+'/2/'+cl + '/train')
        os.makedirs(dataset_Dir+'/2/'+cl + '/train/groundtruth')
        os.makedirs(dataset_Dir+'/2/'+cl + '/test')
        os.makedirs(dataset_Dir + '/2/' + cl + '/test/groundtruth')

if not os.path.exists(dataset_Dir + '/3'):
    os.makedirs(dataset_Dir+'/3')
    for cl in split_3:
        os.makedirs(dataset_Dir+'/3/'+ cl)
        os.makedirs(dataset_Dir+'/3/'+cl + '/train')
        os.makedirs(dataset_Dir+'/3/'+cl + '/train/groundtruth')
        os.makedirs(dataset_Dir+'/3/'+cl + '/test')
        os.makedirs(dataset_Dir + '/3/' + cl + '/test/groundtruth')




for cl in split_0:
    catIds=coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1
    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label=np.array(Image.open(label_path))
        ann_mask = (label==cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/0/' + cl + '/test/groundtruth/'+gt_name,ann_mask)
    print('Split 0/' + cl + '/Test set Finished')

for cl in split_1:
    catIds=coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label=np.array(Image.open(label_path))
        ann_mask = (label==cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/1/' + cl + '/test/groundtruth/'+gt_name,ann_mask)
    print('Split 1/' + cl + '/Test set Finished')

for cl in split_2:
    catIds=coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label=np.array(Image.open(label_path))
        ann_mask = (label==cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/2/' + cl + '/test/groundtruth/'+gt_name,ann_mask)
    print('Split 2/' + cl + '/Test set Finished')

for cl in split_3:
    catIds=coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label=np.array(Image.open(label_path))
        ann_mask = (label==cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/3/' + cl + '/test/groundtruth/'+gt_name,ann_mask)
    print('Split 3/' + cl + '/Test set Finished')

dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
label_dir = 'PATH_TO_TRAINING_LABEL_DIRECTORY'

coco=COCO(annFile)

for cl in split_0:
    catIds = coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label = np.array(Image.open(label_path))
        ann_mask = (label == cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/0/' + cl + '/train/groundtruth/' + gt_name, ann_mask)
    print('Split 0/' + cl + '/Train set Finished')

for cl in split_1:
    catIds = coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label = np.array(Image.open(label_path))
        ann_mask = (label == cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/1/' + cl + '/train/groundtruth/' +gt_name, ann_mask)
    print('Split 1/' + cl + '/Train set Finished')

for cl in split_2:
    catIds = coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label = np.array(Image.open(label_path))
        ann_mask = (label == cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/2/' + cl + '/train/groundtruth/' + gt_name, ann_mask)
    print('Split 2/' + cl + '/Train set Finished')

for cl in split_3:
    catIds = coco.getCatIds(catNms=[cl])
    imgIds = coco.getImgIds(catIds=catIds)
    cat = nms.index(cl) + 1

    for imgid in tqdm(imgIds):
        img = coco.loadImgs(imgid)[0]
        filename = img['file_name']
        gt_name = filename.split('.')[0] + '.png'
        label_path = os.path.join(label_dir, gt_name)
        label = np.array(Image.open(label_path))
        ann_mask = (label == cat)
        if ann_mask.sum() >= 1:
            plt.imsave(dataset_Dir + '/3/' + cl + '/train/groundtruth/' + gt_name, ann_mask)
    print('Split 3/' + cl + '/Train set Finished')
