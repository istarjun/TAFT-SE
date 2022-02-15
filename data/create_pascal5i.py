import os
import shutil


Pascal5i_label_path = 'PATH/TO/Binary_map_aug'
VOC_Image_path = 'PATH/TO/VOC/IMAGES'
Pascal5i_dataset_path = 'PATH/TO/CREATE/PASCAL5i'
class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


split_0 = class_list[0:5]
split_1 = class_list[5:10]
split_2 = class_list[10:15]
split_3 = class_list[15:20]

if not os.path.exists(Pascal5i_dataset_path):
  os.makedirs(Pascal5i_dataset_path)

if not os.path.exists(Pascal5i_dataset_path + '/0'):
    os.makedirs(Pascal5i_dataset_path+'/0')
    for cl in split_0:
        os.makedirs(Pascal5i_dataset_path+'/0/'+cl)
        os.makedirs(Pascal5i_dataset_path+'/0/'+cl + '/train')
        os.makedirs(Pascal5i_dataset_path+'/0/'+cl + '/train/origin')
        os.makedirs(Pascal5i_dataset_path+'/0/'+cl + '/train/groundtruth')
        os.makedirs(Pascal5i_dataset_path+'/0/'+cl + '/test')
        os.makedirs(Pascal5i_dataset_path + '/0/' + cl + '/test/origin')
        os.makedirs(Pascal5i_dataset_path + '/0/' + cl + '/test/groundtruth')

if not os.path.exists(Pascal5i_dataset_path + '/1'):
    os.makedirs(Pascal5i_dataset_path+'/1')
    for cl in split_1:
        os.makedirs(Pascal5i_dataset_path+'/1/'+cl)
        os.makedirs(Pascal5i_dataset_path+'/1/'+cl + '/train')
        os.makedirs(Pascal5i_dataset_path+'/1/'+cl + '/train/origin')
        os.makedirs(Pascal5i_dataset_path+'/1/'+cl + '/train/groundtruth')
        os.makedirs(Pascal5i_dataset_path+'/1/'+cl + '/test')
        os.makedirs(Pascal5i_dataset_path + '/1/' + cl + '/test/origin')
        os.makedirs(Pascal5i_dataset_path + '/1/' + cl + '/test/groundtruth')

if not os.path.exists(Pascal5i_dataset_path + '/2'):
    os.makedirs(Pascal5i_dataset_path+'/2')
    for cl in split_2:
        os.makedirs(Pascal5i_dataset_path+'/2/'+cl)
        os.makedirs(Pascal5i_dataset_path+'/2/'+cl + '/train')
        os.makedirs(Pascal5i_dataset_path+'/2/'+cl + '/train/origin')
        os.makedirs(Pascal5i_dataset_path+'/2/'+cl + '/train/groundtruth')
        os.makedirs(Pascal5i_dataset_path+'/2/'+cl + '/test')
        os.makedirs(Pascal5i_dataset_path + '/2/' + cl + '/test/origin')
        os.makedirs(Pascal5i_dataset_path + '/2/' + cl + '/test/groundtruth')

if not os.path.exists(Pascal5i_dataset_path + '/3'):
    os.makedirs(Pascal5i_dataset_path+'/3')
    for cl in split_3:
        os.makedirs(Pascal5i_dataset_path+'/3/'+ cl)
        os.makedirs(Pascal5i_dataset_path+'/3/'+cl + '/train')
        os.makedirs(Pascal5i_dataset_path+'/3/'+cl + '/train/origin')
        os.makedirs(Pascal5i_dataset_path+'/3/'+cl + '/train/groundtruth')
        os.makedirs(Pascal5i_dataset_path+'/3/'+cl + '/test')
        os.makedirs(Pascal5i_dataset_path + '/3/' + cl + '/test/origin')
        os.makedirs(Pascal5i_dataset_path + '/3/' + cl + '/test/groundtruth')


for i in range(1, 21):
  valid_data = []
  split = (i-1)//5
  with open(Pascal5i_label_path + '/train/' + '%d.txt' % (i)) as txtfile:
    lines = txtfile.readlines()
    for single_line in lines:
      line_data = single_line.replace('\n', ' ').split(' ')
      line_data.remove('')
      valid_data.append(line_data[0])
  for data in valid_data:
    shutil.copy(VOC_Image_path + '/' + data + '.jpg',
                Pascal5i_dataset_path + '/' + str(split) + '/' + class_list[i - 1] + '/train/origin/' + data + '.jpg')
    shutil.copy(str(i) + '/' + data + '.png',
                Pascal5i_dataset_path + '/' + str(split) + '/' + class_list[i - 1] + '/train/groundtruth/' + data + '.png')
  valid_data = []
  with open(Pascal5i_label_path + '/val/' + '%d.txt' % (i)) as txtfile:
    lines = txtfile.readlines()
    split = (i - 1) // 5
    for single_line in lines:
      line_data = single_line.replace('\n', ' ').split(' ')
      line_data.remove('')
      valid_data.append(line_data[0])
  for data in valid_data:
    shutil.copy(VOC_Image_path + '/' + data + '.jpg',
                Pascal5i_dataset_path + '/' + str(split) + '/' + class_list[i - 1] + '/test/origin/' + data + '.jpg')
    shutil.copy(str(i) + '/' + data + '.png',
                Pascal5i_dataset_path + '/' + str(split) + '/' + class_list[i - 1] + '/test/groundtruth/' + data + '.png')
  print('%d th class complete' % i)
