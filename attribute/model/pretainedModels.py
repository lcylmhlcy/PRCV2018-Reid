# -*- coding: UTF-8 -*-

import torchvision.models as models
from torchsummary import summary
import torch.nn as nn

def resnet152_model():
    resnet152 = models.resnet152(pretrained=True)
    resnet152.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    # summary(resnet152.cuda(),(3,224,224))
    # print(resnet152)
    return resnet152

def densenet161_model():
    densenet161 = models.densenet161(pretrained=True)
    densenet161.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
    # summary(densenet161.cuda(),(3,224,224))
    # print(densenet161)
    return densenet161
    
def inception_v3_model():
    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    # summary(inception_v3.cuda(),(3,299,299))
    # print(inception_v3)
    return inception_v3
    
def resnet101_model():
    resnet101 = models.resnet101(pretrained=True)
    resnet101.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    # summary(resnet101.cuda(),(3,224,224))
    # print(resnet101)
    return resnet101
    
def densenet201_model():
    densenet201 = models.densenet201(pretrained=True)
    densenet201.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
    # summary(densenet201.cuda(),(3,224,224))
    # print(densenet201)
    return densenet201
    
def resnet50_model():
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    # summary(resnet50.cuda(),(3,224,224))
    # print(resnet50)
    return resnet50
    
def densenet169_model():
    densenet169 = models.densenet169(pretrained=True)
    densenet169.classifier = nn.Linear(in_features=1664, out_features=2, bias=True)
    # summary(densenet169.cuda(),(3,224,224))
    # print(densenet169)
    return densenet169

def densenet121_model():
    densenet121 = models.densenet121(pretrained=True) 
    densenet121.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    # summary(densenet121.cuda(),(3,224,224)) 
    # print(densenet121)
    return densenet121
    
def vgg19_bn_model():
    vgg19_bn = models.vgg19_bn(pretrained=True)
    vgg19_bn.classifier =  nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=2, bias=True)
    )
    # summary(vgg19_bn.cuda(),(3,224,224))
    # print(vgg19_bn)
    return vgg19_bn
    
def vgg16_bn_model():
    vgg16_bn = models.vgg16_bn(pretrained=True)   
    vgg16_bn.classifier =  nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=2, bias=True)
    )
    # summary(vgg16_bn.cuda(),(3,224,224))
    # print(vgg16_bn)
    return vgg16_bn
    
# resnet152_model()
# densenet161_model()
# inception_v3_model()
# resnet101_model()
# densenet201_model()
# resnet50_model()
# densenet169_model()
# densenet121_model()
# vgg19_bn_model()
# vgg16_bn_model()