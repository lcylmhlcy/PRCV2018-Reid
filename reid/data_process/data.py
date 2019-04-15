#coding:utf-8
#读取每张图片和对应的尺寸大小，train和test图片分别写入train_img_size.txt和test_img_size.txt中

import os
from PIL import Image
import numpy as np
import scipy.io
import torchvision.transforms as transforms
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


train_img_path = '/home/computer/lcy/prcv/data/ReID/training_images/'
test_img_path = '/home/computer/lcy/prcv/data/ReID/test_images/'

train_mat_path = '/home/computer/lcy/prcv/data/ReID/RAP_reid_data.mat' 
data = scipy.io.loadmat(train_mat_path)
data = data['RAP_reid_data']

train_data = data['training_set']
train_data = train_data[0][0]


train_file = open('/home/computer/lcy/prcv/experiment/my/data_process/train_img_size.txt', "a+")

for i in range(13178):
    temp_img_path = train_data[i][0][0]
    print(temp_img_path)
    train_img_label = train_data[i][1][0][0]
    img = Image.open(os.path.join(train_img_path,str(temp_img_path)))
    
    print(img.size)
    print(train_img_label)
    print('--------------------------------------------')
    train_file.write(str(temp_img_path) + '   ' + str(img.size) + '   ' + str(train_img_label) + '\n')
    
train_file.close()
'''
######################################################################################################################
test_data = data['test_set']
test_data = test_data[0][0]


test_file = open('/home/computer/lcy/prcv/experiment/my/data_process/test_img_size.txt', "a+")

for i in range(13460):
    temp_img_path = test_data[i][0][0]
    print(temp_img_path)
    test_img_label = test_data[i][1][0][0]
    img = Image.open(os.path.join(test_img_path,str(temp_img_path)))
    
    print(img.size)
    print(test_img_label)
    print('--------------------------------------------')
    test_file.write(str(temp_img_path) + '   ' + str(img.size) + '   ' + str(test_img_label) + '\n')
    
test_file.close()
'''