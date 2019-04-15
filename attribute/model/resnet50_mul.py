# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv_1 = nn.Conv2d(2048, 1620, 1, 1, 0)
        self.fc = nn.Linear(30, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.conv_1(x)
        x = torch.squeeze(x)
        print(x.shape)
        
        x0 = self.fc(x[0:30]) 
        x1 = self.fc(x[30:60])
        x2 = self.fc(x[60:90])
        x3 = self.fc(x[90:120])
        x4 = self.fc(x[120:150])
        x5 = self.fc(x[150:180])
        x6 = self.fc(x[180:210])
        x7 = self.fc(x[210:240])
        x8 = self.fc(x[240:270])
        x9 = self.fc(x[270:300])
        
        x10 = self.fc(x[300:30]) 
        x11 = self.fc(x[330:60])
        x12 = self.fc(x[360:90])
        x13 = self.fc(x[390:120])
        x14 = self.fc(x[420:150])
        x15 = self.fc(x[450:180])
        x16 = self.fc(x[480:210])
        x17 = self.fc(x[510:240])
        x18 = self.fc(x[540:270])
        x19 = self.fc(x[570:600])       
        x20 = self.fc(x[600:630]) 
        x21 = self.fc(x[630:660])
        x22 = self.fc(x[660:690])
        x23 = self.fc(x[690:720])
        x24 = self.fc(x[720:750])
        x25 = self.fc(x[750:780])
        x26 = self.fc(x[780:810])
        x27 = self.fc(x[810:840])
        x28 = self.fc(x[840:870])
        x29 = self.fc(x[870:900])        
        x30 = self.fc(x[900:930]) 
        x31 = self.fc(x[930:960])
        x32 = self.fc(x[960:990])
        x33 = self.fc(x[990:1020])
        x34 = self.fc(x[1020:1050])
        x35 = self.fc(x[1050:1080])
        x36 = self.fc(x[1080:1110])
        x37 = self.fc(x[1110:1140])
        x38 = self.fc(x[1140:1170])
        x39 = self.fc(x[1170:1200])        
        x40 = self.fc(x[1200:1230]) 
        x41 = self.fc(x[1230:1260])
        x42 = self.fc(x[1260:1290])
        x43 = self.fc(x[1290:1320])
        x44 = self.fc(x[1320:1350])
        x45 = self.fc(x[1350:1380])
        x46 = self.fc(x[1380:1410])
        x47 = self.fc(x[1410:1440])
        x48 = self.fc(x[1440:1470])
        x49 = self.fc(x[1470:1500])        
        x50 = self.fc(x[1500:1530]) 
        x51 = self.fc(x[1530:1560])
        x52 = self.fc(x[1560:1590])
        x53 = self.fc(x[1590:1620])


        return x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,
        x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,
        x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,
        x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,
        x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,
        x50,x51,x52,x53

def resnet50_fc_model():
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_state = model.state_dict()
    resnet50 = models.resnet50(pretrained=True)
    pre_dict =  resnet50.state_dict()
    pre_trained_dict = {k: v for k, v in pre_dict.item() if k in model_state}
    model_state.update(pre_trained_dict)
    model.load_state_dict(model_state)
    
    return model