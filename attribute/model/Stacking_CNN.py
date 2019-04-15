# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .pretainedModels import resnet152_model, densenet161_model, inception_v3_model, resnet101_model, densenet201_model, resnet50_model, densenet169_model, densenet121_model, vgg19_bn_model, vgg16_bn_model


class StackingCNN(nn.Module):  
    def __init__(self):
        super(StackingCNN, self).__init__()  

        self.resnet152 = resnet152_model()
        self.densenet161 = densenet161_model()
        self.resnet101 = resnet101_model()
        self.densenet201 = densenet201_model()
        self.resnet50 = resnet50_model()
        self.densenet169 = densenet169_model()
        self.densenet121 = densenet121_model()
        self.vgg19_bn = vgg19_bn_model()
        self.vgg16_bn = vgg16_bn_model()
        self.fusion = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3,1), stride = 1, padding=0),
                                    nn.Conv2d(8, 16, kernel_size=(3,1), stride = 1, padding=0)
                                    )
        
        self.classifier = nn.Linear(in_features=160, out_features=2, bias=True)

    def forward(self, x):
        x1 = self.resnet152(x)
        x2 = self.densenet161(x)
        # x3 = self.inception_v3(x)
        x3 = self.resnet101(x)
        x4 = self.densenet201(x)
        x5 = self.resnet50(x)
        x6 = self.densenet169(x)
        x7 = self.densenet121(x)
        x8 = self.vgg19_bn(x)
        x9 = self.vgg16_bn(x)
        
        batch = int(x1.shape[0])
        result = []
        for i in range(batch):     
            y = torch.cat((x1[i].unsqueeze(0), x2[i].unsqueeze(0), x3[i].unsqueeze(0), x4[i].unsqueeze(0), x5[i].unsqueeze(0), x6[i].unsqueeze(0), x7[i].unsqueeze(0), x8[i].unsqueeze(0), x9[i].unsqueeze(0)), 0)
            y = y.unsqueeze(0)
            y = y.unsqueeze(0) 
            y = self.fusion(y)
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
            result.append(y)
                    
        del x1, x2, x3, x4, x5, x6, x7, x8, x9, y
        output = torch.cat(result, 0)
        del result
        
        return output