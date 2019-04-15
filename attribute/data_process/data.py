#coding:utf-8
#训练数据读取 train.h5

import os
from PIL import Image
import numpy as np
# np.set_printoptions(threshold='nan') 
import scipy.io
import torchvision.transforms as transforms
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.Resize((384,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )

train_img_path = '/home/computer/lcy/prcv/data/Attributes/training_validation_images/'

train_mat_path = '/home/computer/lcy/prcv/data/Attributes/RAP_attributes_data.mat' 
data = scipy.io.loadmat(train_mat_path)
data = data['RAP_attributes_data']

train_data = data['training_validation_sets'][0][0][0][0]
# print(train_data)

train_img_allname = train_data[0]
# print(train_img_allname)

partition = train_data[4][0][0]
training_index = partition[0][0]
val_index = partition[1][0]
val_index_list = []
for i in range(8317):
    val_temp_index = str(val_index[i])
    val_temp_index = val_temp_index.zfill(5)
    val_index_list.append(val_temp_index)
    
train_select_indexs = train_data[5][0] - 1
train_all_labels = train_data[1]
train_select_labels = np.expand_dims(train_all_labels[:, train_select_indexs[0]], axis=1)
for i in range(1,54):
    select_temp = np.expand_dims(train_all_labels[:, train_select_indexs[i]], axis=1)
    train_select_labels = np.hstack((train_select_labels,select_temp))


training_img = []
val_img = []
training_label = []
val_label = []


for i in range(41585):   
    all_label = train_select_labels[i]
    
    temp_img_path = train_img_allname[i][0][0]
    print(temp_img_path)
    img = Image.open(os.path.join(train_img_path, temp_img_path))
    img = transform(img)
    img = img.numpy()
    img = img.tolist()
       
    _, _, _, img_index = temp_img_path.split('_')
    img_index, _ = img_index.split('.')
    print(img_index)
    
    print(all_label[0])
    if all_label[0] != 2:
        if img_index in val_index_list:
            val_img.append(img)
            val_label.append(all_label)
            print('val')
        else:
            training_img.append(img)
            training_label.append(all_label)
            print('training')
    else:
        print('wrong data!!!!!!!!!!!!!!')

    print('------------------------------------------')
    
training_img = np.asarray(training_img,dtype=np.float32)   
print(training_img.shape)
training_label = np.asarray(training_label,dtype=np.int64)  
print(training_label.shape)  
val_img = np.asarray(val_img,dtype=np.float32)   
print(val_img.shape)
val_label = np.asarray(val_label,dtype=np.int64)  
print(val_label.shape)  


f1 = h5py.File('training_384_128_new.h5','w')
f1['training_img'] = training_img                
f1['training_label'] = training_label
f1.close()

f2 = h5py.File('val_384_128_new.h5','w')
f2['val_img'] = val_img                
f2['val_label'] = val_label
f2.close()