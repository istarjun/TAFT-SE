from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

dataDir = 'PATH_TO_COCO2014_DATASET'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

dataset_Dir = 'PATH_TO_CREATE_TRAINING_LABEL_DIRECTORY'

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
imgIds = coco.getImgIds()
if not os.path.exists(dataset_Dir):
    os.makedirs(dataset_Dir)

for idx, im_id in tqdm(enumerate(imgIds)):
    annIds = coco.getAnnIds(imgIds=im_id, iscrowd=False)
    if len(annIds)==0:
        continue
    anns = coco.loadAnns(annIds)
    image = coco.loadImgs([im_id])[0]
    h, w = image['height'], image['width']
    gt_name = image['file_name'].split('.')[0] + '.png'
    gt = np.zeros((h, w), dtype=np.uint8)
    for ann_idx, ann in enumerate(anns):
        cat = coco.loadCats([ann['category_id']])
        cat = cat[0]['name']
        cat = nms.index(cat) + 1
        ann_mask = coco.annToMask(ann)
        x,y = ann_mask.nonzero()
        gt[x,y]=cat
    target_image = Image.fromarray(gt, mode='P')
    save_gt_path = os.path.join(dataset_Dir, gt_name)
    target_image.save(save_gt_path)

dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

dataset_Dir = 'PATH_TO_CREATE_VALIDATION_LABEL_DIRECTORY'

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
imgIds = coco.getImgIds()
if not os.path.exists(dataset_Dir):
    os.makedirs(dataset_Dir)

for idx, im_id in tqdm(enumerate(imgIds)):
    annIds = coco.getAnnIds(imgIds=im_id, iscrowd=False)
    if len(annIds)==0:
        continue
    anns = coco.loadAnns(annIds)
    image = coco.loadImgs([im_id])[0]
    h, w = image['height'], image['width']
    gt_name = image['file_name'].split('.')[0] + '.png'
    gt = np.zeros((h, w), dtype=np.uint8)
    for ann_idx, ann in enumerate(anns):
        cat = coco.loadCats([ann['category_id']])
        cat = cat[0]['name']
        cat = nms.index(cat) + 1
        ann_mask = coco.annToMask(ann)
        x,y = ann_mask.nonzero()
        gt[x,y]=cat
    target_image = Image.fromarray(gt, mode='P')
    save_gt_path = os.path.join(dataset_Dir, gt_name)
    target_image.save(save_gt_path)
