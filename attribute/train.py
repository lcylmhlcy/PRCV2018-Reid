# -*- coding: UTF-8 -*-

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
from model.resnet50_original import ft_net
from model.dataset import MyDataset
from model.import_data import data_import
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


training_img,training_female,training_ageless16,training_age17_30,training_age31_45,training_age46_60,training_bodyfat,training_bodynormal,training_bodythin,training_customer,training_employee,training_hs_baldhead,training_hs_longhair,training_hs_blackhair,training_hs_hat,training_hs_glasses,training_ub_shirt,training_ub_sweater,training_ub_vest,training_ub_tshirt,training_ub_cotton,training_ub_jacket,training_ub_suitup,training_ub_tight,training_ub_shortsleeve,training_ub_others,training_lb_longtrousers,training_lb_skirt,training_lb_shortskirt,training_lb_dress,training_lb_jeans,training_lb_tighttrousers,training_shoes_leather,training_shoes_sports,training_shoes_boots,training_shoes_cloth,training_shoes_casual,training_shoes_others,training_attachment_backpack,training_attachment_shoulderbag,training_attachment_handbag,training_attachment_box,training_attachment_plasticbag,training_attachment_paperbag,training_attachment_handtrunk,training_attachment_other,training_action_calling,training_action_talking,training_action_gathering,training_action_holding,training_action_pushing,training_action_pulling,training_action_carryingbyarm,training_action_carryingbyhand,training_action_other,val_img,val_female,val_ageless16,val_age17_30,val_age31_45,val_age46_60,val_bodyfat,val_bodynormal,val_bodythin,val_customer,val_employee,val_hs_baldhead,val_hs_longhair,val_hs_blackhair,val_hs_hat,val_hs_glasses,val_ub_shirt,val_ub_sweater,val_ub_vest,val_ub_tshirt,val_ub_cotton,val_ub_jacket,val_ub_suitup,val_ub_tight,val_ub_shortsleeve,val_ub_others,val_lb_longtrousers,val_lb_skirt,val_lb_shortskirt,val_lb_dress,val_lb_jeans,val_lb_tighttrousers,val_shoes_leather,val_shoes_sports,val_shoes_boots,val_shoes_cloth,val_shoes_casual,val_shoes_others,val_attachment_backpack,val_attachment_shoulderbag,val_attachment_handbag,val_attachment_box,val_attachment_plasticbag,training_attachment_paperbag,val_attachment_handtrunk,val_attachment_other,val_action_calling,val_action_talking,val_action_gathering,val_action_holding,val_action_pushing,val_action_pulling,val_action_carryingbyarm,val_action_carryingbyhand,val_action_other = data_import()



## 超参数
batch_size = 96
epochs = 50

train_dataset = MyDataset(training_img,training_female,training_ageless16,training_age17_30,training_age31_45,training_age46_60,training_bodyfat,training_bodynormal,training_bodythin,training_customer,training_employee,training_hs_baldhead,training_hs_longhair,training_hs_blackhair,training_hs_hat,training_hs_glasses,training_ub_shirt,training_ub_sweater,training_ub_vest,training_ub_tshirt,training_ub_cotton,training_ub_jacket,training_ub_suitup,training_ub_tight,training_ub_shortsleeve,training_ub_others,training_lb_longtrousers,training_lb_skirt,training_lb_shortskirt,training_lb_dress,training_lb_jeans,training_lb_tighttrousers,training_shoes_leather,training_shoes_sports,training_shoes_boots,training_shoes_cloth,training_shoes_casual,training_shoes_others,training_attachment_backpack,training_attachment_shoulderbag,training_attachment_handbag,training_attachment_box,training_attachment_plasticbag,training_attachment_paperbag,training_attachment_handtrunk,training_attachment_other,training_action_calling,training_action_talking,training_action_gathering,training_action_holding,training_action_pushing,training_action_pulling,training_action_carryingbyarm,training_action_carryingbyhand,training_action_other)

train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_dataset = MyDataset(val_img,val_female,val_ageless16,val_age17_30,val_age31_45,val_age46_60,val_bodyfat,val_bodynormal,val_bodythin,val_customer,val_employee,val_hs_baldhead,val_hs_longhair,val_hs_blackhair,val_hs_hat,val_hs_glasses,val_ub_shirt,val_ub_sweater,val_ub_vest,val_ub_tshirt,val_ub_cotton,val_ub_jacket,val_ub_suitup,val_ub_tight,val_ub_shortsleeve,val_ub_others,val_lb_longtrousers,val_lb_skirt,val_lb_shortskirt,val_lb_dress,val_lb_jeans,val_lb_tighttrousers,val_shoes_leather,val_shoes_sports,val_shoes_boots,val_shoes_cloth,val_shoes_casual,val_shoes_others,val_attachment_backpack,val_attachment_shoulderbag,val_attachment_handbag,val_attachment_box,val_attachment_plasticbag,training_attachment_paperbag,val_attachment_handtrunk,val_attachment_other,val_action_calling,val_action_talking,val_action_gathering,val_action_holding,val_action_pushing,val_action_pulling,val_action_carryingbyarm,val_action_carryingbyhand,val_action_other)

val_loader = dataf.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = ft_net()
model.load_state_dict(torch.load('./saved_model/6_75_model.pkl'))
model = model.cuda()
# summary(model, (3, 384, 128))

criterion = nn.CrossEntropyLoss()

param_groups = [{'params': model.resnet.parameters(), 'lr': 1e-6},
                {'params': model.fc1.parameters(), 'lr': 1e-5}]
optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=0.005)

# print('-------------------------------------------------------------------------')

for epoch in range(epochs):
    # 训练模式
    model.train()

    # 初始化每个epoch的总损失
    train_loss = 0.
    train_acc0 = 0
    train_acc1 = 0
    train_acc2 = 0
    train_acc3 = 0
    train_acc4 = 0
    train_acc5 = 0
    train_acc6 = 0
    train_acc7 = 0
    train_acc8 = 0
    train_acc9 = 0
    train_acc10 = 0
    train_acc11 = 0
    train_acc12 = 0
    train_acc13 = 0
    train_acc14 = 0
    train_acc15 = 0
    train_acc16 = 0
    train_acc17 = 0
    train_acc18 = 0
    train_acc19 = 0
    train_acc20 = 0
    train_acc21 = 0
    train_acc22 = 0
    train_acc23 = 0
    train_acc24 = 0
    train_acc25 = 0
    train_acc26 = 0
    train_acc27 = 0
    train_acc28 = 0
    train_acc29 = 0
    train_acc30 = 0
    train_acc31 = 0
    train_acc32 = 0
    train_acc33 = 0
    train_acc34 = 0
    train_acc35 = 0
    train_acc36 = 0
    train_acc37 = 0
    train_acc38 = 0
    train_acc39 = 0
    train_acc40 = 0
    train_acc41 = 0
    train_acc42 = 0
    train_acc43 = 0
    train_acc44 = 0
    train_acc45 = 0
    train_acc46 = 0
    train_acc47 = 0
    train_acc48 = 0
    train_acc49 = 0
    train_acc50 = 0
    train_acc51 = 0
    train_acc52 = 0
    train_acc53 = 0


    # 每次输入barch_idx个数据
    for batch_idx, (training_img,training_female,training_ageless16,training_age17_30,training_age31_45,training_age46_60,training_bodyfat,training_bodynormal,training_bodythin,training_customer,training_employee,training_hs_baldhead,training_hs_longhair,training_hs_blackhair,training_hs_hat,training_hs_glasses,training_ub_shirt,training_ub_sweater,training_ub_vest,training_ub_tshirt,training_ub_cotton,training_ub_jacket,training_ub_suitup,training_ub_tight,training_ub_shortsleeve,training_ub_others,training_lb_longtrousers,training_lb_skirt,training_lb_shortskirt,training_lb_dress,training_lb_jeans,training_lb_tighttrousers,training_shoes_leather,training_shoes_sports,training_shoes_boots,training_shoes_cloth,training_shoes_casual,training_shoes_others,training_attachment_backpack,training_attachment_shoulderbag,training_attachment_handbag,training_attachment_box,training_attachment_plasticbag,training_attachment_paperbag,training_attachment_handtrunk,training_attachment_other,training_action_calling,training_action_talking,training_action_gathering,training_action_holding,training_action_pushing,training_action_pulling,training_action_carryingbyarm,training_action_carryingbyhand,training_action_other) in enumerate(train_loader):

        training_img = training_img.cuda()

        training_female,training_ageless16,training_age17_30,training_age31_45,training_age46_60,training_bodyfat,training_bodynormal,training_bodythin,training_customer,training_employee,training_hs_baldhead,training_hs_longhair,training_hs_blackhair,training_hs_hat,training_hs_glasses,training_ub_shirt,training_ub_sweater,training_ub_vest,training_ub_tshirt,training_ub_cotton,training_ub_jacket,training_ub_suitup,training_ub_tight,training_ub_shortsleeve,training_ub_others,training_lb_longtrousers,training_lb_skirt,training_lb_shortskirt,training_lb_dress,training_lb_jeans,training_lb_tighttrousers,training_shoes_leather,training_shoes_sports,training_shoes_boots,training_shoes_cloth,training_shoes_casual,training_shoes_others,training_attachment_backpack,training_attachment_shoulderbag,training_attachment_handbag,training_attachment_box,training_attachment_plasticbag,training_attachment_paperbag,training_attachment_handtrunk,training_attachment_other,training_action_calling,training_action_talking,training_action_gathering,training_action_holding,training_action_pushing,training_action_pulling,training_action_carryingbyarm,training_action_carryingbyhand,training_action_other = training_female.cuda(),training_ageless16.cuda(),training_age17_30.cuda(),training_age31_45.cuda(),training_age46_60.cuda(),training_bodyfat.cuda(),training_bodynormal.cuda(),training_bodythin.cuda(),training_customer.cuda(),training_employee.cuda(),training_hs_baldhead.cuda(),training_hs_longhair.cuda(),training_hs_blackhair.cuda(),training_hs_hat.cuda(),training_hs_glasses.cuda(),training_ub_shirt.cuda(),training_ub_sweater.cuda(),training_ub_vest.cuda(),training_ub_tshirt.cuda(),training_ub_cotton.cuda(),training_ub_jacket.cuda(),training_ub_suitup.cuda(),training_ub_tight.cuda(),training_ub_shortsleeve.cuda(),training_ub_others.cuda(),training_lb_longtrousers.cuda(),training_lb_skirt.cuda(),training_lb_shortskirt.cuda(),training_lb_dress.cuda(),training_lb_jeans.cuda(),training_lb_tighttrousers.cuda(),training_shoes_leather.cuda(),training_shoes_sports.cuda(),training_shoes_boots.cuda(),training_shoes_cloth.cuda(),training_shoes_casual.cuda(),training_shoes_others.cuda(),training_attachment_backpack.cuda(),training_attachment_shoulderbag.cuda(),training_attachment_handbag.cuda(),training_attachment_box.cuda(),training_attachment_plasticbag.cuda(),training_attachment_paperbag.cuda(),training_attachment_handtrunk.cuda(),training_attachment_other.cuda(),training_action_calling.cuda(),training_action_talking.cuda(),training_action_gathering.cuda(),training_action_holding.cuda(),training_action_pushing.cuda(),training_action_pulling.cuda(),training_action_carryingbyarm.cuda(),training_action_carryingbyhand.cuda(),training_action_other.cuda()

        optimizer.zero_grad()

        output = model(training_img)
        
        loss0 = criterion(output[0], training_female)       
        acc0 = (torch.max(output[0], 1)[1] == training_female).sum()
        train_acc0 += int(acc0)
        
        loss1 = criterion(output[1], training_ageless16)
        acc1 = (torch.max(output[1], 1)[1] == training_ageless16).sum()
        train_acc1 += int(acc1)
        
        loss2 = criterion(output[2], training_age17_30)        
        acc2 = (torch.max(output[2], 1)[1] == training_age17_30).sum()
        train_acc2 += int(acc2)
        
        loss3 = criterion(output[3], training_age31_45)        
        acc3 = (torch.max(output[3], 1)[1] == training_age31_45).sum()
        train_acc3 += int(acc3)
        
        loss4 = criterion(output[4], training_age46_60)        
        acc4 = (torch.max(output[4], 1)[1] == training_age46_60).sum()
        train_acc4 += int(acc4)
        
        loss5 = criterion(output[5], training_bodyfat)        
        acc5 = (torch.max(output[5], 1)[1] == training_bodyfat).sum()
        train_acc5 += int(acc5)
        
        loss6 = criterion(output[6], training_bodynormal)        
        acc6 = (torch.max(output[6], 1)[1] == training_bodynormal).sum()
        train_acc6 += int(acc6)
        
        loss7 = criterion(output[7], training_bodythin)        
        acc7 = (torch.max(output[7], 1)[1] == training_bodythin).sum()
        train_acc7 += int(acc7)
        
        loss8 = criterion(output[8], training_customer)        
        acc8 = (torch.max(output[8], 1)[1] == training_customer).sum()
        train_acc8 += int(acc8)
        
        loss9 = criterion(output[9], training_employee)        
        acc9 = (torch.max(output[9], 1)[1] == training_employee).sum()
        train_acc9 += int(acc9)
        
        loss10 = criterion(output[10], training_hs_baldhead)        
        acc10 = (torch.max(output[10], 1)[1] == training_hs_baldhead).sum()
        train_acc10 += int(acc10)
        
        loss11 = criterion(output[11], training_hs_longhair)        
        acc11 = (torch.max(output[11], 1)[1] == training_hs_longhair).sum()
        train_acc11 += int(acc11)
        
        loss12 = criterion(output[12], training_hs_blackhair)        
        acc12 = (torch.max(output[12], 1)[1] == training_hs_blackhair).sum()
        train_acc12 += int(acc12)  
        
        loss13 = criterion(output[13], training_hs_hat)        
        acc13 = (torch.max(output[13], 1)[1] == training_hs_hat).sum()
        train_acc13 += int(acc13)
        
        loss14 = criterion(output[14], training_hs_glasses)        
        acc14 = (torch.max(output[14], 1)[1] == training_hs_glasses).sum()
        train_acc14 += int(acc14)
        
        loss15 = criterion(output[15], training_ub_shirt)        
        acc15 = (torch.max(output[15], 1)[1] == training_ub_shirt).sum()
        train_acc15 += int(acc15)
        
        loss16 = criterion(output[16], training_ub_sweater)        
        acc16 = (torch.max(output[16], 1)[1] == training_ub_sweater).sum()
        train_acc16 += int(acc16)
        
        loss17 = criterion(output[17], training_ub_vest)        
        acc17 = (torch.max(output[17], 1)[1] == training_ub_vest).sum()
        train_acc17 += int(acc17)
        
        loss18 = criterion(output[18], training_ub_tshirt)        
        acc18 = (torch.max(output[18], 1)[1] == training_ub_tshirt).sum()
        train_acc18 += int(acc18)
        
        loss19 = criterion(output[19], training_ub_cotton)        
        acc19 = (torch.max(output[19], 1)[1] == training_ub_cotton).sum()
        train_acc19 += int(acc19)
        
        loss20 = criterion(output[20], training_ub_jacket)        
        acc20 = (torch.max(output[20], 1)[1] == training_ub_jacket).sum()
        train_acc20 += int(acc20)
        
        loss21 = criterion(output[21], training_ub_suitup)        
        acc21 = (torch.max(output[21], 1)[1] == training_ub_suitup).sum()
        train_acc21 += int(acc21)
        
        loss22 = criterion(output[22], training_ub_tight)        
        acc22 = (torch.max(output[22], 1)[1] == training_ub_tight).sum()
        train_acc22 += int(acc22)
        
        loss23 = criterion(output[23], training_ub_shortsleeve)        
        acc23 = (torch.max(output[23], 1)[1] == training_ub_shortsleeve).sum()
        train_acc23 += int(acc23)
        
        loss24 = criterion(output[24], training_ub_others)        
        acc24 = (torch.max(output[24], 1)[1] == training_ub_others).sum()
        train_acc24 += int(acc24)
        
        loss25 = criterion(output[25], training_lb_longtrousers)        
        acc25 = (torch.max(output[25], 1)[1] == training_lb_longtrousers).sum()
        train_acc25 += int(acc25)
        
        loss26 = criterion(output[26], training_lb_skirt)        
        acc26 = (torch.max(output[26], 1)[1] == training_lb_skirt).sum()
        train_acc26 += int(acc26)
        
        loss27 = criterion(output[27], training_lb_shortskirt)        
        acc27 = (torch.max(output[27], 1)[1] == training_lb_shortskirt).sum()
        train_acc27 += int(acc27)
        
        loss28 = criterion(output[28], training_lb_dress)        
        acc28 = (torch.max(output[28], 1)[1] == training_lb_dress).sum()
        train_acc28 += int(acc28) 
        
        loss29 = criterion(output[29], training_lb_jeans)        
        acc29 = (torch.max(output[29], 1)[1] == training_lb_jeans).sum()
        train_acc29 += int(acc29)
        
        loss30 = criterion(output[30], training_lb_tighttrousers)        
        acc30 = (torch.max(output[30], 1)[1] == training_lb_tighttrousers).sum()
        train_acc30 += int(acc30)
        
        loss31 = criterion(output[31], training_shoes_leather)        
        acc31 = (torch.max(output[31], 1)[1] == training_shoes_leather).sum()
        train_acc31 += int(acc31)
        
        loss32 = criterion(output[32], training_shoes_sports)        
        acc32 = (torch.max(output[32], 1)[1] == training_shoes_sports).sum()
        train_acc32 += int(acc32)
        
        loss33 = criterion(output[33], training_shoes_boots)        
        acc33 = (torch.max(output[33], 1)[1] == training_shoes_boots).sum()
        train_acc33 += int(acc33)
        
        loss34 = criterion(output[34], training_shoes_cloth)        
        acc34 = (torch.max(output[34], 1)[1] == training_shoes_cloth).sum()
        train_acc34 += int(acc34)
        
        loss35 = criterion(output[35], training_shoes_casual)        
        acc35 = (torch.max(output[35], 1)[1] == training_shoes_casual).sum()
        train_acc35 += int(acc35)
        
        loss36 = criterion(output[36], training_shoes_others)        
        acc36 = (torch.max(output[36], 1)[1] == training_female).sum()
        train_acc36 += int(acc36)
        
        loss37 = criterion(output[37], training_attachment_backpack)        
        acc37 = (torch.max(output[37], 1)[1] == training_attachment_backpack).sum()
        train_acc37 += int(acc37)
        
        loss38 = criterion(output[38], training_attachment_shoulderbag)        
        acc38 = (torch.max(output[38], 1)[1] == training_attachment_shoulderbag).sum()
        train_acc38 += int(acc38)
        
        loss39 = criterion(output[39], training_attachment_handbag)        
        acc39 = (torch.max(output[39], 1)[1] == training_attachment_handbag).sum()
        train_acc39 += int(acc39)
        
        loss40 = criterion(output[40], training_attachment_box)        
        acc40 = (torch.max(output[40], 1)[1] == training_attachment_box).sum()
        train_acc40 += int(acc40)
        
        loss41 = criterion(output[41], training_attachment_plasticbag)        
        acc41 = (torch.max(output[41], 1)[1] == training_attachment_plasticbag).sum()
        train_acc41 += int(acc41)
        
        loss42 = criterion(output[42], training_attachment_paperbag)        
        acc42 = (torch.max(output[42], 1)[1] == training_attachment_paperbag).sum()
        train_acc42 += int(acc42)
        
        loss43 = criterion(output[43], training_attachment_handtrunk)        
        acc43 = (torch.max(output[43], 1)[1] == training_attachment_handtrunk).sum()
        train_acc43 += int(acc43)
        
        loss44 = criterion(output[44], training_attachment_other)        
        acc44 = (torch.max(output[44], 1)[1] == training_attachment_other).sum()
        train_acc44 += int(acc44)
        
        loss45 = criterion(output[45], training_action_calling)        
        acc45 = (torch.max(output[45], 1)[1] == training_action_calling).sum()
        train_acc45 += int(acc45)
        
        loss46 = criterion(output[46], training_action_talking)        
        acc46 = (torch.max(output[46], 1)[1] == training_action_talking).sum()
        train_acc46 += int(acc46)
        
        loss47 = criterion(output[47], training_action_gathering)        
        acc47 = (torch.max(output[47], 1)[1] == training_action_gathering).sum()
        train_acc47 += int(acc47)
        
        loss48 = criterion(output[48], training_action_holding)        
        acc48 = (torch.max(output[48], 1)[1] == training_action_holding).sum()
        train_acc48 += int(acc48)
        
        loss49 = criterion(output[49], training_action_pushing)        
        acc49 = (torch.max(output[49], 1)[1] == training_action_pushing).sum()
        train_acc49 += int(acc49)
        
        loss50 = criterion(output[50], training_action_pulling)        
        acc50 = (torch.max(output[50], 1)[1] == training_action_pulling).sum()
        train_acc50 += int(acc50)
        
        loss51 = criterion(output[51], training_action_carryingbyarm)        
        acc51 = (torch.max(output[51], 1)[1] == training_action_carryingbyarm).sum()
        train_acc51 += int(acc51)
        
        loss52 = criterion(output[52], training_action_carryingbyhand)        
        acc52 = (torch.max(output[52], 1)[1] == training_action_carryingbyhand).sum()
        train_acc52 += int(acc52)
        
        loss53 = criterion(output[53], training_action_other)        
        acc53 = (torch.max(output[53], 1)[1] == training_action_other).sum()
        train_acc53 += int(acc53)
        
        
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16 + loss17 + loss18 + loss19 + loss20 + loss21 + loss22 + loss23 + loss24 + loss25 + loss26 + loss27 + loss28 + loss29 + loss30 + loss31 + loss32 + loss33 + loss34 + loss35 + loss36 + loss37 + loss38 + loss39 + loss40 + loss41 + loss42 + loss43 + loss44 + loss45 + loss46 + loss47 + loss48 + loss49 + loss50 + loss51 + loss52 + loss53
        
        train_loss += float(loss)
        
        # 回传并更新梯度
        loss.backward()
        optimizer.step()

        
        # 输出结果
        print('Train Epoch: {}  Batch:{} [{}/{} ({:.0f}%)]\tLoss_id: {:.6f}'.format(
            epoch + 1,
            batch_idx+1,
            batch_idx * len(training_img),
            len(train_dataset),
            100. * batch_idx * len(training_img) / len(train_dataset),
            loss.item() / batch_size)
        )
        print('----------------------------------------------------------------------------------------------------------')
        
        model_name = './saved_model/' + str(epoch+1) + '_' + str(batch_idx) + '_model.pkl'
        torch.save(model.state_dict(), model_name)
        

    # print('############################')
    print('############################################################################################################')
    print('Train Epoch: [{}\{}]\tLoss_id: {:.6f}'.format(epoch + 1, epochs, train_loss / len(train_dataset)))
    print('training_female:                 {:.0f}%'.format(100 * train_acc0 / len(train_dataset)))
    print('training_ageless16:              {:.0f}%'.format(100 * train_acc1 / len(train_dataset)))
    print('training_age17_30:               {:.0f}%'.format(100 * train_acc2 / len(train_dataset)))
    print('training_age31_45:               {:.0f}%'.format(100 * train_acc3 / len(train_dataset)))
    print('training_age46_60:               {:.0f}%'.format(100 * train_acc4 / len(train_dataset)))
    print('training_bodyfat:                {:.0f}%'.format(100 * train_acc5 / len(train_dataset)))
    print('training_bodynormal:             {:.0f}%'.format(100 * train_acc6 / len(train_dataset)))
    print('training_bodythin:               {:.0f}%'.format(100 * train_acc7 / len(train_dataset)))
    print('training_customer:               {:.0f}%'.format(100 * train_acc8 / len(train_dataset)))
    print('training_employee:               {:.0f}%'.format(100 * train_acc9 / len(train_dataset)))
    print('training_hs_baldhead:            {:.0f}%'.format(100 * train_acc10 / len(train_dataset)))
    print('training_hs_longhair:            {:.0f}%'.format(100 * train_acc11 / len(train_dataset)))
    print('training_hs_blackhair:           {:.0f}%'.format(100 * train_acc12 / len(train_dataset)))
    print('training_hs_hat:                 {:.0f}%'.format(100 * train_acc13 / len(train_dataset)))
    print('training_hs_glasses:             {:.0f}%'.format(100 * train_acc14 / len(train_dataset)))
    print('training_ub_shirt:               {:.0f}%'.format(100 * train_acc15 / len(train_dataset)))
    print('training_ub_sweater:             {:.0f}%'.format(100 * train_acc16 / len(train_dataset)))
    print('training_ub_vest:                {:.0f}%'.format(100 * train_acc17 / len(train_dataset)))
    print('training_ub_tshirt:              {:.0f}%'.format(100 * train_acc18 / len(train_dataset)))
    print('training_ub_cotton:              {:.0f}%'.format(100 * train_acc19 / len(train_dataset)))
    print('training_ub_jacket:              {:.0f}%'.format(100 * train_acc20 / len(train_dataset)))
    print('training_ub_suitup:              {:.0f}%'.format(100 * train_acc21 / len(train_dataset)))
    print('training_ub_tight:               {:.0f}%'.format(100 * train_acc22 / len(train_dataset)))
    print('training_ub_shortsleeve:         {:.0f}%'.format(100 * train_acc23 / len(train_dataset)))
    print('training_ub_others:              {:.0f}%'.format(100 * train_acc24 / len(train_dataset)))
    print('training_lb_longtrousers:        {:.0f}%'.format(100 * train_acc25 / len(train_dataset)))
    print('training_lb_skirt:               {:.0f}%'.format(100 * train_acc26 / len(train_dataset)))
    print('training_lb_shortskirt:          {:.0f}%'.format(100 * train_acc27 / len(train_dataset)))
    print('training_lb_dress:               {:.0f}%'.format(100 * train_acc28 / len(train_dataset)))
    print('training_lb_jeans:               {:.0f}%'.format(100 * train_acc29 / len(train_dataset)))
    print('training_lb_tighttrousers:       {:.0f}%'.format(100 * train_acc30 / len(train_dataset)))
    print('training_shoes_leather:          {:.0f}%'.format(100 * train_acc31 / len(train_dataset)))
    print('training_shoes_sports:           {:.0f}%'.format(100 * train_acc32 / len(train_dataset)))
    print('training_shoes_boots:            {:.0f}%'.format(100 * train_acc33 / len(train_dataset)))
    print('training_shoes_cloth:            {:.0f}%'.format(100 * train_acc34 / len(train_dataset)))
    print('training_shoes_casual:           {:.0f}%'.format(100 * train_acc35 / len(train_dataset)))
    print('training_shoes_others:           {:.0f}%'.format(100 * train_acc36 / len(train_dataset)))
    print('training_attachment_backpack:    {:.0f}%'.format(100 * train_acc37 / len(train_dataset)))
    print('training_attachment_shoulderbag: {:.0f}%'.format(100 * train_acc38 / len(train_dataset)))
    print('training_attachment_handbag:     {:.0f}%'.format(100 * train_acc39 / len(train_dataset)))
    print('training_attachment_box:         {:.0f}%'.format(100 * train_acc40 / len(train_dataset)))
    print('training_attachment_plasticbag:  {:.0f}%'.format(100 * train_acc41 / len(train_dataset)))
    print('training_attachment_paperbag:    {:.0f}%'.format(100 * train_acc42 / len(train_dataset)))
    print('training_attachment_handtrunk:   {:.0f}%'.format(100 * train_acc43 / len(train_dataset)))
    print('training_attachment_other:       {:.0f}%'.format(100 * train_acc44 / len(train_dataset)))
    print('training_action_calling:         {:.0f}%'.format(100 * train_acc45 / len(train_dataset)))
    print('training_action_talking:         {:.0f}%'.format(100 * train_acc46 / len(train_dataset)))
    print('training_action_gathering:       {:.0f}%'.format(100 * train_acc47 / len(train_dataset)))
    print('training_action_holding:         {:.0f}%'.format(100 * train_acc48 / len(train_dataset)))
    print('training_action_pushing:         {:.0f}%'.format(100 * train_acc49 / len(train_dataset)))
    print('training_action_pulling:         {:.0f}%'.format(100 * train_acc50 / len(train_dataset)))
    print('training_action_carryingbyarm:   {:.0f}%'.format(100 * train_acc51 / len(train_dataset)))
    print('training_action_carryingbyhand:  {:.0f}%'.format(100 * train_acc52 / len(train_dataset)))
    print('training_action_other:           {:.0f}%'.format(100 * train_acc53 / len(train_dataset)))
    print('############################################################################################################')
    # print('############################')
    print('----------------------------------------------------------------------------------------------------------')
    
    '''
    model.eval()
    '''
        

