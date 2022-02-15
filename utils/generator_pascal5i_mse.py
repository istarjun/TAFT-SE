import numpy as np
import os
import random
from PIL import Image 

import utils.custom_transforms as tr 
from torchvision import transforms 

class pascal5i_generator_test(object):
    def __init__(self, path_dir, n_epoch, n_way, n_sample, split_number, multi_scale,train=True, n_workers=1, xp=np):
        super(pascal5i_generator_test, self).__init__()
        self.path_dir = path_dir
        self.n_epoch = n_epoch
        self.n_way = n_way
        self.n_sample=n_sample
        self.split_number = split_number 
        self.train=train
        self.xp = xp
        self.num_iter = 0
        self.n_workers = n_workers
        self.split_list = [0,1,2,3]
        self.multi_scale = multi_scale
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
            images_1 = []
            labels_1 = []
            images_2 = []
            labels_2 = []
            images_3 = []
            labels_3 = []
            split = self.split_number
            cl_ind = (self.num_iter-1)//(self.n_epoch//5)
            cl = self.class_list[5*split:5*(split+1)][cl_ind]
            
            class_dir = self.path_dir + str(split) + '/' + cl + '/test'
            sample_list = os.listdir(class_dir+'/origin')    
            sample_ind = random.sample(sample_list, self.n_sample)
            
            scale_factor = self.multi_scale[0]
            image_size = int(512*scale_factor)
            
            for j in sample_ind:
                image = Image.open(class_dir + '/origin/'+str(j))
                j_label = j[:-4] + '.png'
                label = Image.open(class_dir + '/groundtruth/'+str(j_label))
                label = label.convert("L")
                sample = {'image':image, 'label':label}
                sample_tr = self.transform(sample, image_size)
                image_resize = sample_tr['image']
                label_resize = sample_tr['label']
                images_1.append(image_resize)
                labels_1.append(label_resize)
    
                    
            images_1 = np.stack(images_1)
            labels_1 = np.stack(labels_1)
            
            scale_factor = self.multi_scale[1]
            image_size = int(512*scale_factor)
            
            for j in sample_ind:
                image = Image.open(class_dir + '/origin/'+str(j))
                j_label = j[:-4] + '.png'
                label = Image.open(class_dir + '/groundtruth/'+str(j_label))
                label = label.convert("L")
                sample = {'image':image, 'label':label}
                sample_tr = self.transform(sample, image_size)
                image_resize = sample_tr['image']
                label_resize = sample_tr['label']
                images_2.append(image_resize)
                labels_2.append(label_resize)
    
                    
            images_2 = np.stack(images_2)
            labels_2 = np.stack(labels_2)
            
            scale_factor = self.multi_scale[2]
            image_size = int(512*scale_factor)
            
            for j in sample_ind:
                image = Image.open(class_dir + '/origin/'+str(j))
                j_label = j[:-4] + '.png'
                label = Image.open(class_dir + '/groundtruth/'+str(j_label))
                label = label.convert("L")
                sample = {'image':image, 'label':label}
                sample_tr = self.transform(sample, image_size)
                image_resize = sample_tr['image']
                label_resize = sample_tr['label']
                images_3.append(image_resize)
                labels_3.append(label_resize)
    
                    
            images_3 = np.stack(images_3)
            labels_3 = np.stack(labels_3)
            
            
            return (self.num_iter - 1), (images_1, labels_1, images_2, labels_2, images_3, labels_3), cl_ind
        else:
            raise StopIteration()
    
    


        