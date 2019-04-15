# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet = model_ft       
        self.fc1 = nn.Linear(2048, 2)


    def forward(self, x):
        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)
        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)
        attr = self.resnet.avgpool(y)
        attr = attr.view(attr.size(0), -1)
        out_attr1 = F.softmax(self.fc1(attr), dim=1)
        out_attr2 = F.softmax(self.fc1(attr), dim=1)
        out_attr3 = F.softmax(self.fc1(attr), dim=1)
        out_attr4 = F.softmax(self.fc1(attr), dim=1)
        out_attr5 = F.softmax(self.fc1(attr), dim=1)
        out_attr6 = F.softmax(self.fc1(attr), dim=1)
        out_attr7 = F.softmax(self.fc1(attr), dim=1)
        out_attr8 = F.softmax(self.fc1(attr), dim=1)
        out_attr9 = F.softmax(self.fc1(attr), dim=1)
        out_attr10 = F.softmax(self.fc1(attr), dim=1)
        out_attr11 = F.softmax(self.fc1(attr), dim=1)
        out_attr12 = F.softmax(self.fc1(attr), dim=1)
        out_attr13 = F.softmax(self.fc1(attr), dim=1)
        out_attr14 = F.softmax(self.fc1(attr), dim=1)
        out_attr15 = F.softmax(self.fc1(attr), dim=1)
        out_attr16 = F.softmax(self.fc1(attr), dim=1)
        out_attr17 = F.softmax(self.fc1(attr), dim=1)
        out_attr18 = F.softmax(self.fc1(attr), dim=1)
        out_attr19 = F.softmax(self.fc1(attr), dim=1)
        out_attr20 = F.softmax(self.fc1(attr), dim=1)
        out_attr21 = F.softmax(self.fc1(attr), dim=1)
        out_attr22 = F.softmax(self.fc1(attr), dim=1)
        out_attr23 = F.softmax(self.fc1(attr), dim=1)
        out_attr24 = F.softmax(self.fc1(attr), dim=1)
        out_attr25 = F.softmax(self.fc1(attr), dim=1)
        out_attr26 = F.softmax(self.fc1(attr), dim=1)
        out_attr27 = F.softmax(self.fc1(attr), dim=1)
        out_attr28 = F.softmax(self.fc1(attr), dim=1)
        out_attr29 = F.softmax(self.fc1(attr), dim=1)
        out_attr30 = F.softmax(self.fc1(attr), dim=1)
        out_attr31 = F.softmax(self.fc1(attr), dim=1)
        out_attr32 = F.softmax(self.fc1(attr), dim=1)
        out_attr33 = F.softmax(self.fc1(attr), dim=1)
        out_attr34 = F.softmax(self.fc1(attr), dim=1)
        out_attr35 = F.softmax(self.fc1(attr), dim=1)
        out_attr36 = F.softmax(self.fc1(attr), dim=1)
        out_attr37 = F.softmax(self.fc1(attr), dim=1)
        out_attr38 = F.softmax(self.fc1(attr), dim=1)
        out_attr39 = F.softmax(self.fc1(attr), dim=1)
        out_attr40 = F.softmax(self.fc1(attr), dim=1)
        out_attr41 = F.softmax(self.fc1(attr), dim=1)
        out_attr42 = F.softmax(self.fc1(attr), dim=1)
        out_attr43 = F.softmax(self.fc1(attr), dim=1)
        out_attr44 = F.softmax(self.fc1(attr), dim=1)
        out_attr45 = F.softmax(self.fc1(attr), dim=1)
        out_attr46 = F.softmax(self.fc1(attr), dim=1)
        out_attr47 = F.softmax(self.fc1(attr), dim=1)
        out_attr48 = F.softmax(self.fc1(attr), dim=1)
        out_attr49 = F.softmax(self.fc1(attr), dim=1)
        out_attr50 = F.softmax(self.fc1(attr), dim=1)
        out_attr51 = F.softmax(self.fc1(attr), dim=1)
        out_attr52 = F.softmax(self.fc1(attr), dim=1)
        out_attr53 = F.softmax(self.fc1(attr), dim=1)
        out_attr54 = F.softmax(self.fc1(attr), dim=1)


        return out_attr1,out_attr2,out_attr3,out_attr4,out_attr5,out_attr6,out_attr7,out_attr8,out_attr9,out_attr10,out_attr11,out_attr12,out_attr13,out_attr14,out_attr15,out_attr16,out_attr17,out_attr18,out_attr19,out_attr20,out_attr21,out_attr22,out_attr23,out_attr24,out_attr25,out_attr26,out_attr27,out_attr28,out_attr29,out_attr30,out_attr31,out_attr32,out_attr33,out_attr34,out_attr35,out_attr36,out_attr37,out_attr38,out_attr39,out_attr40,out_attr41,out_attr42,out_attr43,out_attr44,out_attr45,out_attr46,out_attr47,out_attr48,out_attr49,out_attr50, out_attr51,out_attr52,out_attr53,out_attr54