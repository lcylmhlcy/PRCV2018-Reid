# -*- coding: UTF-8 -*-
# training_384_128_33268.h5   keys: training_img training_label
# val_384_128_8317.h5         keys: val_img val_label

import numpy as np
# np.set_printoptions(threshold='nan')
import h5py
import torch
from torch import nn
import torchvision.models as models
import torch.utils.data as dataf
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary
from model.Stacking_CNN import StackingCNN
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


f1 = h5py.File('/home/computer/lcy/prcv/experiment/attribute/data_process/training_224_224_33268.h5','r')
training_img = np.asarray(f1['training_img'])      
training_label = np.asarray(f1['training_label'])
training_label = training_label[:,0]
f1.close()

f2 = h5py.File('/home/computer/lcy/prcv/experiment/attribute/data_process/val_384_128_8317.h5','r')
val_img = np.asarray(f2['val_img'])       
val_label = np.asarray(f2['val_label'])
val_label = val_label[:,0]
f2.close()

batch_size = 3
epochs = 1000

# class MyDataset(dataf.Dataset):
    # def __init__(self, img_data, label_list):
        # self.img_data = img_data
        # print(img_data.shape)
        # print(label_list.shape)
        # self.label_list = label_list

    # def __getitem__(self, index):
        # label_temp = self.label_list[:,0]
        # return self.img_data[index], label_temp[index]

    # def __len__(self):
        # return self.img_data.size(0)
        
training_img = torch.from_numpy(training_img)
training_label = torch.from_numpy(training_label)

train_dataset = dataf.TensorDataset(training_img, training_label)
train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_img = torch.from_numpy(val_img)
val_label = torch.from_numpy(val_label)

val_dataset = dataf.TensorDataset(val_img, val_label)
val_loader = dataf.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = StackingCNN()
model = model.cuda()
# summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD([
                {'params': model.resnet152.parameters()},
                {'params': model.densenet161.parameters()},
                {'params': model.resnet101.parameters()},
                {'params': model.densenet201.parameters()},
                {'params': model.resnet50.parameters()},
                {'params': model.densenet169.parameters()},
                {'params': model.densenet121.parameters()},
                {'params': model.vgg19_bn.parameters()},
                {'params': model.vgg16_bn.parameters()},
                {'params': model.fusion.parameters(), 'lr': 1e-3},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-4, momentum=0.9, weight_decay=0.005)
# print('-------------------------------------------------------------------------')

for epoch in range(epochs):

    model.train()

    train_loss = 0.
    train_acc = 0

    for batch_idx, (train_img, train_label) in enumerate(train_loader):
       
        train_img = train_img.cuda()       
        train_label = train_label.cuda()

        optimizer.zero_grad()

        out = model(train_img)

        loss = criterion(out, train_label)
        train_loss += float(loss)

        pred = torch.max(out, 1)[1]
        train_correct = (pred == train_label).sum()
        train_acc += int(train_correct)
        
        del pred, train_correct
       
        loss.backward()
        optimizer.step()

        print('Train Epoch: {}  [{}/{} ({:.0f}%)]\tLoss_id: {:.6f}'.format(
            epoch + 1,
            batch_idx * len(train_img),
            len(train_dataset),
            100. * batch_idx / len(train_dataset),
            loss.item() / batch_size)
        )

        print('----------------------------------------------------------------------------------------------------------')

    
    print('############################')
    print('############################')
    print('Train Epoch: [{}\{}]\tLoss_id: {:.6f}\tAcc:{:.2f}%'.format(epoch + 1, epochs, train_loss / len(train_dataset), 100 * train_acc / len(train_dataset)))
    print('############################')
    print('############################')

    model_name = './saved_model/0/' + str(epoch+1) + '_model.pkl'
    torch.save(model.state_dict(), model_name)
    
    if epoch == 5:
        model.eval()
        
        val_acc = 0
        for batch_idx1, (val_img, val_label) in enumerate(val_loader):       
            val_img = val_img.cuda()       
            out1 = model(val_img)
            pred1 = torch.max(out1, 1)[1]
            val_correct = (pred1 == val_label).sum()
            val_acc += int(val_correct)
        print('########################################')
        print('val_acc:  {:.2f}%'.format(100 * val_acc / len(val_dataset)))

