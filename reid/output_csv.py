#coding:utf-8
# 10_0_model.pkl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.PCBModel import PCBModel
import csv

transform = transforms.Compose([
        transforms.Resize((384,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )

test_img_path = '/home/computer/lcy/prcv/data/ReID/test_images/'
csv_path = '/home/computer/lcy/prcv/experiment/reid/csv/3/'

   
def TestAndQuery_namelist():   
    test_mat_path = '/home/computer/lcy/prcv/data/ReID/RAP_reid_data.mat' 
    data = scipy.io.loadmat(test_mat_path)
    data = data['RAP_reid_data']
    test_data = data['test_set']
    test_data = test_data[0][0]
    # print(test_data.shape)
    # print(test_data[0][0][0])
       
    all_test_name = []
    for i in range(test_data.shape[0]):
        temp_test_name = test_data[i][0][0]
        all_test_name.append(temp_test_name)
    
    
    all_query_name = []
    for line in open("/home/computer/lcy/prcv/data/ReID/query_test_image_name.txt"):
        line=line.strip('\n')
        all_query_name.append(line)
        
    # print(len(all_test_name))
    # print(len(all_query_name))
    
    return all_query_name, all_test_name
    
def feature_extractor(all_query_name, all_test_name):
    model = PCBModel()
    model.load_state_dict(torch.load('./saved_model/19_48_model.pkl'))
    model.cuda()
    model.eval()
    
    all_test_feature = []
    all_query_feature = []
    
    for img_name in all_test_name:
        # print(img_name)
        img = Image.open(os.path.join(test_img_path,img_name))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            local_feat_list, logits_list = model(img)
            feat = [lf.data.cpu().numpy() for lf in local_feat_list]
            feat = np.concatenate(feat, axis=1)
            feat = np.squeeze(feat)        
        all_test_feature.append(feat)            
        # print('----------------------------------------------')
    
    for img_name in all_query_name:
        # print(img_name)
        img = Image.open(os.path.join(test_img_path,img_name))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            local_feat_list, logits_list = model(img)
            feat = [lf.data.cpu().numpy() for lf in local_feat_list]
            feat = np.concatenate(feat, axis=1)
            feat = np.squeeze(feat)        
        # print('query')
        all_query_feature.append(feat)
            
        # print('----------------------------------------------')
            
    all_query_feature = np.asarray(all_query_feature, dtype = np.float32)
    all_test_feature = np.asarray(all_test_feature, dtype = np.float32)
    
    # print(all_query_feature.shape)
    # print(all_test_feature.shape)
    
    f1 = h5py.File(os.path.join(csv_path,'feature.h5'),'w')
    f1['all_query_feature'] = all_query_feature                
    f1['all_test_feature'] = all_test_feature
    f1.close()
    
    return all_query_feature, all_test_feature
    
    
def load_featureh5(featureh5_name):
    f2 = h5py.File(featureh5_name,'r')
    all_query_feature = f2['all_query_feature'][()]               
    all_test_feature = f2['all_test_feature'][()]
    f2.close()
    
    return all_query_feature, all_test_feature
   

def compute_dist(array1, array2):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    Returns:
    numpy array with shape [m1, m2]
    """
    
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    
    # print(dist.shape)
    
    f3 = h5py.File(os.path.join(csv_path,'dist.h5'),'w')
    f3['dist'] = dist                
    f3.close()
    
    return dist    

def load_disth5(disth5_name):
    f4 = h5py.File(disth5_name,'r')
    dist = f4['dist'][()]               
    f4.close()
    
    return dist
    

def csv_output(all_query_name, all_test_name, feature_dist):
    query_len = len(all_query_name)
    test_len = len(all_test_name)
    
       
    with open(os.path.join(csv_path,'reid.csv'),'w', newline='') as csvFile:
        wr = csv.writer(csvFile)
        
        # first_row = []
        # first_row.append('<query_index>')
        # for i1 in range(test_len):
        #     first_row.append('<image_index_'+str(i1+1)+'>')
        #     first_row.append('<confidence_'+str(i1+1)+'>')
        # wr.writerow(first_row)
        
        for i in range(query_len):
            temp_row = []
            temp_query_name = i #all_query_name[i]
            print(temp_query_name)
            temp_row.append(str(temp_query_name))
            temp_dist_list = feature_dist[i]
            temp_index_list = np.argsort(temp_dist_list)
            print(temp_index_list)
            for index in temp_index_list:
                temp_row.append(str(index)) #all_test_name[index]
                temp_row.append(str(temp_dist_list[index]))
            
            wr.writerow(temp_row)
            
            print('-----------------------------------------------------')
      
if __name__=='__main__':
    all_query_name, all_test_name = TestAndQuery_namelist()
    all_query_feature, all_test_feature = feature_extractor(all_query_name, all_test_name)
    # all_query_feature, all_test_feature = load_featureh5(os.path.join(csv_path,'feature.h5'))
    feature_dist = compute_dist(all_query_feature, all_test_feature)
    # feature_dist = load_disth5(os.path.join(csv_path,'dist.h5')) 
    csv_output(all_query_name, all_test_name, feature_dist)