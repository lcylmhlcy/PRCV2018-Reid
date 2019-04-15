# coding:utf-8
# /home/computer/lcy/prcv/experiment/attribute/saved_model/all/1_model.pkl

import os
from PIL import Image
import numpy as np
import scipy.io
import torchvision.transforms as transforms
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.resnet50_original import ft_net
import torch
import csv

transform = transforms.Compose([
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)
test_img_path = '/home/computer/lcy/prcv/data/Attributes/test_images/'
csv_path = '/home/computer/lcy/prcv/experiment/attribute/csv/'

all_query_attrname = []
for line in open("/home/computer/lcy/prcv/data/Attributes/attr_query_index.txt"):
    line = line.strip()
    line = line.strip('\n')
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int64)
    # print(line)
    all_query_attrname.append(line)
all_query_attrname = np.asarray(all_query_attrname)
csvFile = open('csv/query_results.csv', 'w', newline = '')
writer = csv.writer(csvFile)
m = len(all_query_attrname)
test_mat_path = '/home/computer/lcy/prcv/data/Attributes/RAP_attributes_data.mat'
data = scipy.io.loadmat(test_mat_path)
data = data['RAP_attributes_data']
test_data = data['test_set'][0][0]
all_test_name = []
# print(all_test_name)
print(len(test_data))
print(m)
for i in range(len(test_data)):
    temp_test_name = test_data[i][0][0]
    all_test_name.append(temp_test_name)
# print(all_test_name)
for i in range(m):
    output = str(all_query_attrname[i])
    csvFile.write(output)
    # writer.writerow(f)
    # csvFile.write(f)
    # csvFile.write()
    writer.writerow(all_test_name)
    # csvFile.write(all_test_name)
    # csvFile.write('\n')
csvFile.close()
# for i in range(1):
#      for j in range(3):
#          temp_test_name = test_data[i][0][0]
#          print(temp_test_name)
#          print('\n')
# for i in range(len(test_data)):
#     for j in range(len(test_data)):
#         temp_test_name = test_data[i][0][0]
#         all_test_name = temp_test_name
#     csvFile.write(all_test_name)
#     csvFile.write(' ')
# 读取query_attribute.txt文件

# print(all_query_attrname)
# print(all_query_attrname[0][0])
# print(all_query_attrname[204][0])
# print(all_query_attrname[204][1])
# print(all_query_attrname[204][2])
# print(all_query_attrname[204][3])


# for i in range(m):


    # csvFile.write(all_test_name)
    # output1 = str(all_query_attrname[i][k])
    # output = str(all_query_attrname[i])
    # output = output.strip('[')
    # output = output.strip(']')
    # output = output.strip()
    # output = output.replace(',', ' ')

    # output = str(all_query_attrname[i])
    # csvFile.write(output)
    # writer.writerow('This is two row.')
    # csvFile.write('This is two row.')
    # csvFile.write('\n')
    # writer.writerow(output.append(output1))
    # writer.writerow(all_query_attrname[j])


# with open(os.path.join(csv_path, 'query_results.csv'), 'w', newline='') as csvFile:
#     w = csv.writer(csvFile)

# with open(os.path.join(csv_path, 'query_results.csv'), 'w', newline='') as csvFile:
# wr = csv.writer(csvFile)
# def Query_namelist():
#
#     all_query_attrname = []
#     for line in open("/home/computer/lcy/prcv/data/Attributes/attr_query_index.txt"):
#         line = line.strip()
#         line = line.strip('\n')
#         line = line.split(' ')
#         line = np.asarray(line, dtype=np.int64)
#         print(line)
#         all_query_attrname.append(line)
#     print(all_query_attrname)
#     print(len(all_query_attrname))
#     return all_query_attrname


# def csv(all_query_attrname, all_test_name, all_test_feature):
#     male = np.expand_dims(1 - all_test_feature[:, 0], axis=0)
#     male = male.T
#     all_test_feature = np.hstack((male, all_test_feature))
#     test_len = all_test_feature.shape[0]
#     print(all_test_feature.shape)
#     with open(os.path.join(csv_path, 'query_results.csv'), 'w', newline='') as csvFile:
#         wr = csv.writer(csvFile)

def predict_result(all_test_name):
    model = ft_net()
    model.load_state_dict(torch.load('./saved_model/1_354_model.pkl'))
    model.cuda()
    model.eval()

    all_test_feature = []

    
    for index, img_name in enumerate(all_test_name):
        print(img_name)
        img = Image.open(os.path.join(test_img_path, img_name))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
            # print(output)
            prediction = []
            for attr in output:
                
                # print(attr)
                attr1 = attr.cpu().detach().numpy()
                # attr2 = attr1[0][1]
                # print(attr2)
                # prediction.append(attr2)
            prediction = np.asarray(prediction, dtype=np.float32)
            # print(prediction)

        all_test_feature.append(prediction)
        print(prediction)
        print('---------------------------------------------------------------')

    all_test_feature = np.asarray(all_test_feature, dtype=np.float32)
    # print(all_test_feature.shape)

    f1 = h5py.File(os.path.join(csv_path, 'feature.h5'), 'w')
    f1['all_test_feature'] = all_test_feature
    f1.close()

    return all_test_feature


if __name__ == '__main__':
    # all_query_attrname, all_test_name = TestAndQuery_namelist()
    all_test_feature = predict_result(all_test_name)
    # all_test_feature = load_featureh5(os.path.join(csv_path,'feature.h5'))
    # csv1(all_test_name, all_test_feature)
