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
    tmp = torch.where(input<=5, Zt, input)
    output = torch.where((6<=tmp)&(tmp<=20), tmp-5, tmp)
  elif split == 1:
    tmp = torch.where((6<=input)&(input<=10), Zt, input)
    output = torch.where((11<=tmp)&(tmp<=20), tmp-5, tmp)
  elif split == 2:
    tmp = torch.where((11<=input)&(input<=15), Zt, input)
    output = torch.where((15<=tmp)&(tmp<=20), tmp-5, tmp)
  elif split == 3:
    output = torch.where((16<=input)&(input<=20), Zt, input)
  else:
    raise Exception("Invalid split")

  return output


class pascal5i_generator(object):
    def __init__(self, path_dir, aux_label_dir, n_epoch, n_way, n_sample, split_number, train=True, n_workers=1, xp=np):
        super(pascal5i_generator, self).__init__()
        self.path_dir = path_dir
        self.aux_label_dir = aux_label_dir
        self.n_epoch = n_epoch
        self.n_way = n_way
        self.n_sample=n_sample
        self.split_number = split_number 
        self.train=train
        self.xp = xp
        self.num_iter = 0
        self.n_workers = n_workers
        self.split_list = [0,1,2,3]
        self.split_list.remove(self.split_number)
        self.class_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
                           'cat','chair','cow','diningtable','dog','horse','motorbike',
                           'person','pottedplant','sheep','sofa','train','tvmonitor']
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    def transform(self, sample, size):
        composed_transforms = transforms.Compose([
                tr.FixedResize(size=size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                tr.ToTensor()])
        return composed_transforms(sample)
    
    def next(self):
        if self.num_iter < self.n_epoch:
            self.num_iter += 1
            images = []
            labels = []
            labels_aux = []

            if self.train:
                split = random.sample(self.split_list,1)[0]
                for j in range(self.n_workers):
                    cl = random.sample(self.class_list[5*split:5*(split+1)], 1)[0]
                    
                    class_dir = self.path_dir + str(split) + '/' + cl + '/train'
                    label_dir = self.aux_label_dir + '/train/'
                    sample_list = os.listdir(class_dir+'/origin')    
                    sample_ind = random.sample(sample_list, self.n_sample)
                    for j in sample_ind:
                        image = Image.open(class_dir + '/origin/'+str(j))
                        j_label = j[:-4] + '.png'
                        label = Image.open(class_dir + '/groundtruth/'+str(j_label))
                        label = label.convert("L")
                        label_aux = Image.open(label_dir+str(j_label))
                        label_aux = label_aux.convert("P")
                        sample = {'image':image, 'label':label, 'label_aux':label_aux}
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
                return (self.num_iter - 1), (images, labels,labels_aux)
            else:
                split = self.split_number
                cl_ind = (self.num_iter-1)//(self.n_epoch//5)
                cl = self.class_list[5*split:5*(split+1)][cl_ind]
            
                class_dir = self.path_dir + str(split) + '/' + cl + '/test'
                label_dir = self.aux_label_dir + '/test/'

                sample_list = os.listdir(class_dir+'/origin')    
                sample_ind = random.sample(sample_list, self.n_sample)
                for j in sample_ind:
                    image = Image.open(class_dir + '/origin/'+str(j))
                    j_label = j[:-4] + '.png'
                    label = Image.open(class_dir + '/groundtruth/'+str(j_label))
                    label = label.convert("L")
                    label_aux = Image.open(label_dir + str(j_label))
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
    
    


        