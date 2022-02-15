import numpy as np
import os
import random
from PIL import Image

import utils.custom_transforms_aux as tr
from torchvision import transforms
import torch

def filter_label(split, input):
  Zt = torch.zeros_like(input)
  if split == 0 :
    tmp = torch.where(input%4 ==1, Zt, input)
    tmp = torch.where(tmp == 0, Zt, tmp - (tmp-1)//4-1)
    output = tmp
  elif split == 1:
    tmp = torch.where(input%4 ==2, Zt, input)
    tmp = torch.where(tmp <=1, tmp, tmp - (tmp-2)//4-1)
    output = tmp
  elif split == 2:
    tmp = torch.where(input%4 ==3, Zt, input)
    tmp = torch.where(tmp <=2, tmp, tmp - (tmp-3)//4-1)
    output = tmp
  elif split == 3:
    tmp = torch.where(input%4 ==0, Zt, input)
    tmp = torch.where(tmp <=3, tmp, tmp - (tmp-4)//4-1)
    output = tmp
  else:
    raise Exception("Invalid split")

  return output


class coco20i_generator(object):
    def __init__(self, img_dir, label_dir, aux_label_dir, n_epoch, n_way, n_sample, split_number, train=True, n_workers=1, xp=np):
        super(coco20i_generator, self).__init__()
        self.img_dir = img_dir
        self.label_dir= label_dir
        self.aux_label_dir = aux_label_dir
        self.n_epoch = n_epoch
        self.n_way = n_way
        self.n_sample = n_sample
        self.split_number = split_number
        self.train = train
        self.xp = xp
        self.num_iter = 0
        self.n_workers = n_workers
        self.split_list = [0, 1, 2, 3]
        self.split_list.remove(self.split_number)
        self.class_list = ['person','airplane','boat','parking meter','dog','elephant','backpack','suitcase','sports ball','skateboard','wine glass', 'spoon', 'sandwich','hot dog', 'chair','dining table', 'mouse', 'microwave','refrigerator','scissors',
                           'bicycle','bus','traffic light','bench','horse','bear','umbrella','frisbee','kite','surfboard','cup','bowl','orange','pizza','couch','toilet','remote','oven','book','teddy bear',
                           'car','train','fire hydrant','bird','sheep','zebra','handbag','skis','baseball bat','tennis racket','fork','banana','broccoli','donut','potted plant','tv','keyboard','toaster','clock','hair drier',
                           'motorcycle','truck','stop sign','cat','cow','giraffe','tie','snowboard','baseball glove','bottle','knife','apple','carrot','cake','bed','laptop','cell phone','sink','vase','toothbrush']

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def transform(self, sample, size):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def next(self):
        if self.num_iter < self.n_epoch:
            self.num_iter += 1
            images = []
            labels = []
            labels_aux = []

            if self.train:
                split = random.sample(self.split_list, 1)[0]
                for j in range(self.n_workers):
                    cl = random.sample(self.class_list[20 * split:20 * (split + 1)], 1)[0]
                    class_dir = self.label_dir + str(split) + '/' + cl + '/train'
                    label_dir = self.aux_label_dir + '/train/'
                    sample_list = os.listdir(class_dir + '/groundtruth')
                    sample_ind = random.sample(sample_list, self.n_sample)
                    for j in sample_ind:
                        label = Image.open(class_dir + '/groundtruth/' + str(j))
                        label = label.convert("L")
                        label_aux = Image.open(label_dir + str(j))
                        j_img = j[:-4] + '.jpg'
                        image = Image.open(self.img_dir + 'train2014/' + str(j_img))
                        image = image.convert("RGB")
                        sample = {'image': image, 'label': label, 'label_aux':label_aux}
                        sample_tr = self.transform(sample, 512)
                        image_resize = sample_tr['image']
                        label_resize = sample_tr['label']
                        label_aux_resize = sample_tr['label_aux']
                        label_aux_resize = filter_label(self.split_number, label_aux_resize)
                        images.append(image_resize)
                        labels.append(label_resize)
                        labels_aux.append(label_aux_resize)

                images = np.stack(images)
                labels = np.stack(labels)
                labels_aux = np.stack(labels_aux)

                return (self.num_iter - 1), (images, labels, labels_aux)
            else:
                split = self.split_number
                cl_ind = (self.num_iter - 1) // (self.n_epoch // 20)
                cl = self.class_list[20 * split:20 * (split + 1)][cl_ind]

                class_dir = self.label_dir + str(split) + '/' + cl + '/test'
                label_dir = self.aux_label_dir + '/val/'
                sample_list = os.listdir(class_dir + '/groundtruth')
                sample_ind = random.sample(sample_list, self.n_sample)
                for j in sample_ind:
                    label = Image.open(class_dir + '/groundtruth/' + str(j))
                    label = label.convert("L")
                    label_aux = Image.open(label_dir + str(j))
                    j_img = j[:-4] + '.jpg'
                    image = Image.open(self.img_dir + 'val2014/' + str(j_img))
                    image = image.convert("RGB")
                    sample = {'image': image, 'label': label, 'label_aux': label_aux}
                    sample_tr = self.transform(sample, 512)
                    image_resize = sample_tr['image']
                    label_resize = sample_tr['label']
                    images.append(image_resize)
                    labels.append(label_resize)

                images = np.stack(images)
                labels = np.stack(labels)
                return (self.num_iter - 1), (images, labels), cl_ind
        else:
            raise StopIteration()




