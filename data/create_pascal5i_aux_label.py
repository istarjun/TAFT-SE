import os
import scipy.io
import scipy
import shutil

from PIL import Image

dataset_path = 'PATH/TO/SBD/DATASET'
VOC_label_path = 'PATH/TO/VOC/DATASET/LABELS'

Pascal5i_dataset_path = 'PATH/TO/PASCAL_5i/DATASET'

converted_dataset_path = 'PATH/TO/CREATE/CONVERTED/DATASET'
VOC_SBD_label_path = 'PATH/TO/CREATE/VOC+SBD/LABEL/DATASET'


datalist = os.listdir(dataset_path + '/cls')

if not os.path.exists(converted_dataset_path):
  os.makedirs(converted_dataset_path)

for data in datalist:
  target = scipy.io.loadmat(dataset_path + '/cls/' + data)
  target_array = target["GTcls"][0]['Segmentation'][0]
  target_image = Image.fromarray(target_array, mode='P')

  target_image.save(converted_dataset_path  + data[:-4] + '.png')

VOC_file_list = os.listdir(VOC_label_path)
SBD_file_list = os.listdir(converted_dataset_path)

if not os.path.exists(VOC_SBD_label_path):
  os.makedirs(VOC_SBD_label_path)
  os.makedirs(VOC_SBD_label_path + '/train')
  os.makedirs(VOC_SBD_label_path + '/test')


train_file_list=[]
test_file_list = []
for i in [0,1,2,3]:
    class_list = os.listdir(Pascal5i_dataset_path + '/' + str(i))
    for cl in class_list:
        class_directory = Pascal5i_dataset_path + '/' + str(i) +'/' + cl + '/train/groundtruth'
        data_list = os.listdir(class_directory)
        for data in data_list:
            if data not in train_file_list:
                train_file_list.append(data)
        class_directory = Pascal5i_dataset_path + '/' + str(i) + '/' + cl + '/test/groundtruth'
        data_list = os.listdir(class_directory)
        for data in data_list:
          if data not in test_file_list:
            test_file_list.append(data)

for file in train_file_list:
    if file in VOC_file_list:
      shutil.copy(VOC_label_path + '/' + data + '.png',
                  VOC_SBD_label_path + '/train/' + data + '.png')
    elif file in SBD_file_list:
      shutil.copy(converted_dataset_path + '/' + data + '.png',
                  VOC_SBD_label_path + '/train/' + data + '.png')


for file in test_file_list:
    if file in VOC_file_list:
      shutil.copy(VOC_label_path + '/' + data + '.png',
                  VOC_SBD_label_path + '/test/' + data + '.png')
    elif file in SBD_file_list:
      shutil.copy(converted_dataset_path + '/' + data + '.png',
                  VOC_SBD_label_path + '/test/' + data + '.png')