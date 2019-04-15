# coding:utf-8
# /home/computer/lcy/prcv/experiment/attribute/saved_model/all/1_model.pkl

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


def TestAndQuery_namelist():
    test_mat_path = '/home/computer/lcy/prcv/data/Attributes/RAP_attributes_data.mat'
    data = scipy.io.loadmat(test_mat_path)
    data = data['RAP_attributes_data']

    # test_data = data['test_set'][0][0]
    # print(len(test_data))

    # all_test_name = []
    # for i in range(len(test_data)):
    #     temp_test_name = test_data[i][0][0]
    #     # print(temp_test_name)
    #     all_test_name.append(temp_test_name)

    all_query_attrname = []
    for line in open("/home/computer/lcy/prcv/data/Attributes/attr_query_index.txt"):
        line = line.strip()
        line = line.strip('\n')
        line = line.split(' ')
        line = np.asarray(line, dtype=np.int64)
        # print(line)

        all_query_attrname.append(line)

    # print(all_query_attrname)

    # print(len(all_test_name))
    print(len(all_query_attrname))

    return all_query_attrname


def csv(all_query_attrname, all_test_name, all_test_feature):
    male = np.expand_dims(1 - all_test_feature[:, 0], axis=0)
    male = male.T
    all_test_feature = np.hstack((male, all_test_feature))
    test_len = all_test_feature.shape[0]
    print(all_test_feature.shape)
    with open(os.path.join(csv_path, 'query_results.csv'), 'w', newline='') as csvFile:
        wr = csv.writer(csvFile)



# def predict_query_result(all_query_attrname):
#     model = ft_net
#     model.load_state_dict(torch.load('./saved_model/1_354_model.pkl'))
#     model.cuda()
#     model.eval()
#
#     return all_query_attrname

# def predict_result(all_test_name):
#     model = ft_net()
#     model.load_state_dict(torch.load('./saved_model/1_354_model.pkl'))
#     model.cuda()
#     model.eval()
#
#     all_test_feature = []
#
#     for index, img_name in enumerate(all_test_name):
#         print(img_name)
#         img = Image.open(os.path.join(test_img_path, img_name))
#         img = transform(img)
#         img = img.unsqueeze(0)
#         img = img.cuda()
#         with torch.no_grad():
#             output = model(img)
#             # print(output)
#             prediction = []
#             for attr in output:
#                 # print(attr)
#                 attr1 = attr.cpu().detach().numpy()
#                 attr2 = attr1[0][1]
#                 # print(attr2)
#                 prediction.append(attr2)
#             prediction = np.asarray(prediction, dtype=np.float32)
#             # print(prediction)
#
#         all_test_feature.append(prediction)
#         print('---------------------------------------------------------------')
#
#     all_test_feature = np.asarray(all_test_feature, dtype=np.float32)
#     # print(all_test_feature.shape)
#
#     f1 = h5py.File(os.path.join(csv_path, 'feature.h5'), 'w')
#     f1['all_test_feature'] = all_test_feature
#     f1.close()
#
#     return all_test_feature

# def load_featureh5(featureh5_name):
#     f2 = h5py.File(featureh5_name, 'r')
#     all_test_feature = f2['all_test_feature'][()]
#     f2.close()
#
#     return all_test_feature


# def csv1(all_test_name, all_test_feature):
#     # print(all_test_feature)
#
#     male = 1 - all_test_feature[:, 0]
#     male = np.expand_dims(male, axis=0)
#     male = male.T
#     all_test_feature = np.hstack((male, all_test_feature))
#     test_len = all_test_feature.shape[0]
#     # print(all_test_feature.shape)
#
#
#     with open(os.path.join(csv_path, 'attr_recognition.csv'), 'w', newline='') as csvFile:
#         wr = csv.writer(csvFile)
#
#         first_row = []
#         first_row.append('<image_index>')
#         for i1 in range(55):
#             first_row.append('<confidence_' + str(i1) + '>')
#         wr.writerow(first_row)
#
#         for index, temp_test_name in enumerate(all_test_name):
#             temp_row = []
#             print(temp_test_name)
#             temp_row.append(str(temp_test_name))
#
#             for i in range(55):
#                 temp_attr_predict = all_test_feature[index][i]
#                 temp_row.append(str(temp_attr_predict))
#
#             print(len(temp_row))
#             wr.writerow(temp_row)
#             print('-----------------------------------------------------')
#

if __name__ == '__main__':
    all_query_attrname = TestAndQuery_namelist()
    # all_test_feature = predict_result(all_test_name)
    # all_test_feature = load_featureh5(os.path.join(csv_path,'feature.h5'))
    # csv1(all_test_name, all_test_feature)
