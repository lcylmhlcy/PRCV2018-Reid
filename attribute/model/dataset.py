# -*- coding: UTF-8 -*-

import torch
import torch.utils.data as dataf

class MyDataset(dataf.Dataset):

    def __init__(self, data_tensor, tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8, tensor9, tensor10, tensor11, tensor12, tensor13, tensor14, tensor15, tensor16, tensor17, tensor18, tensor19, tensor20, tensor21, tensor22, tensor23, tensor24, tensor25, tensor26, tensor27, tensor28, tensor29, tensor30, tensor31, tensor32, tensor33, tensor34, tensor35, tensor36, tensor37, tensor38, tensor39, tensor40, tensor41, tensor42, tensor43, tensor44, tensor45, tensor46, tensor47, tensor48, tensor49, tensor50, tensor51, tensor52, tensor53, tensor54):

        self.data_tensor = data_tensor
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3
        self.tensor4 = tensor4
        self.tensor5 = tensor5
        self.tensor6 = tensor6
        self.tensor7 = tensor7
        self.tensor8 = tensor8
        self.tensor9 = tensor9
        self.tensor10 = tensor10
        self.tensor11 = tensor11
        self.tensor12 = tensor12
        self.tensor13 = tensor13
        self.tensor14 = tensor14
        self.tensor15 = tensor15
        self.tensor16 = tensor16
        self.tensor17 = tensor17
        self.tensor18 = tensor18
        self.tensor19 = tensor19
        self.tensor20 = tensor20
        self.tensor21 = tensor21
        self.tensor22 = tensor22
        self.tensor23 = tensor23
        self.tensor24 = tensor24
        self.tensor25 = tensor25
        self.tensor26 = tensor26
        self.tensor27 = tensor27
        self.tensor28 = tensor28
        self.tensor29 = tensor29
        self.tensor30 = tensor30
        self.tensor31 = tensor31
        self.tensor32 = tensor32
        self.tensor33 = tensor33
        self.tensor34 = tensor34
        self.tensor35 = tensor35
        self.tensor36 = tensor36
        self.tensor37 = tensor37
        self.tensor38 = tensor38
        self.tensor39 = tensor39
        self.tensor40 = tensor40
        self.tensor41 = tensor41
        self.tensor42 = tensor42
        self.tensor43 = tensor43
        self.tensor44 = tensor44
        self.tensor45 = tensor45
        self.tensor46 = tensor46
        self.tensor47 = tensor47
        self.tensor48 = tensor48
        self.tensor49 = tensor49
        self.tensor50 = tensor50
        self.tensor51 = tensor51
        self.tensor52 = tensor52
        self.tensor53 = tensor53
        self.tensor54 = tensor54


    def __getitem__(self, index):
        return self.data_tensor[index], self.tensor1[index], self.tensor2[index], self.tensor3[index], self.tensor4[index], self.tensor5[index], self.tensor6[index], self.tensor7[index], self.tensor8[index], self.tensor9[index], self.tensor10[index], self.tensor11[index], self.tensor12[index], self.tensor13[index], self.tensor14[index], self.tensor15[index], self.tensor16[index], self.tensor17[index], self.tensor18[index], self.tensor19[index], self.tensor20[index], self.tensor21[index], self.tensor22[index], self.tensor23[index], self.tensor24[index], self.tensor25[index], self.tensor26[index], self.tensor27[index], self.tensor28[index], self.tensor29[index], self.tensor30[index], self.tensor31[index], self.tensor32[index], self.tensor33[index], self.tensor34[index], self.tensor35[index], self.tensor36[index], self.tensor37[index], self.tensor38[index], self.tensor39[index], self.tensor40[index], self.tensor41[index], self.tensor42[index], self.tensor43[index], self.tensor44[index], self.tensor45[index], self.tensor46[index], self.tensor47[index], self.tensor48[index], self.tensor49[index], self.tensor50[index], self.tensor51[index], self.tensor52[index], self.tensor53[index], self.tensor54[index]

    def __len__(self):
        return self.data_tensor.size(0)