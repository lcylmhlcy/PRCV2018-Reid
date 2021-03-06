# -*- coding: UTF-8 -*-

import numpy as np
# np.set_printoptions(threshold='nan')
import h5py
import torch

def data_import():
    f1 = h5py.File('/home/computer/lcy/prcv/experiment/attribute/data_process/training_384_128_new.h5','r')
    training_img = np.asarray(f1['training_img'])      
    training_label = np.asarray(f1['training_label'])
    f1.close()

    f2 = h5py.File('/home/computer/lcy/prcv/experiment/attribute/data_process/val_384_128_new.h5','r')
    val_img = np.asarray(f2['val_img'])       
    val_label = np.asarray(f2['val_label'])
    f2.close()

    training_img           = torch.from_numpy(training_img)
    training_female                 = torch.from_numpy(training_label[:,0])
    training_ageless16              = torch.from_numpy(training_label[:,1])
    training_age17_30               = torch.from_numpy(training_label[:,2])
    training_age31_45               = torch.from_numpy(training_label[:,3])
    training_age46_60               = torch.from_numpy(training_label[:,4])
    training_bodyfat                = torch.from_numpy(training_label[:,5])
    training_bodynormal             = torch.from_numpy(training_label[:,6])
    training_bodythin               = torch.from_numpy(training_label[:,7])
    training_customer               = torch.from_numpy(training_label[:,8])
    training_employee               = torch.from_numpy(training_label[:,9])
    training_hs_baldhead            = torch.from_numpy(training_label[:,10])
    training_hs_longhair            = torch.from_numpy(training_label[:,11])
    training_hs_blackhair           = torch.from_numpy(training_label[:,12])
    training_hs_hat                 = torch.from_numpy(training_label[:,13])
    training_hs_glasses             = torch.from_numpy(training_label[:,14])
    training_ub_shirt               = torch.from_numpy(training_label[:,15])
    training_ub_sweater             = torch.from_numpy(training_label[:,16])
    training_ub_vest                = torch.from_numpy(training_label[:,17])
    training_ub_tshirt              = torch.from_numpy(training_label[:,18])
    training_ub_cotton              = torch.from_numpy(training_label[:,19])
    training_ub_jacket              = torch.from_numpy(training_label[:,20])
    training_ub_suitup              = torch.from_numpy(training_label[:,21])
    training_ub_tight               = torch.from_numpy(training_label[:,22])
    training_ub_shortsleeve         = torch.from_numpy(training_label[:,23])
    training_ub_others              = torch.from_numpy(training_label[:,24])
    training_lb_longtrousers        = torch.from_numpy(training_label[:,25])
    training_lb_skirt               = torch.from_numpy(training_label[:,26])
    training_lb_shortskirt          = torch.from_numpy(training_label[:,27])
    training_lb_dress               = torch.from_numpy(training_label[:,28])
    training_lb_jeans               = torch.from_numpy(training_label[:,29])
    training_lb_tighttrousers       = torch.from_numpy(training_label[:,30])
    training_shoes_leather          = torch.from_numpy(training_label[:,31])
    training_shoes_sports           = torch.from_numpy(training_label[:,32])
    training_shoes_boots            = torch.from_numpy(training_label[:,33])
    training_shoes_cloth            = torch.from_numpy(training_label[:,34])
    training_shoes_casual           = torch.from_numpy(training_label[:,35])
    training_shoes_others           = torch.from_numpy(training_label[:,36])
    training_attachment_backpack    = torch.from_numpy(training_label[:,37])
    training_attachment_shoulderbag = torch.from_numpy(training_label[:,38])
    training_attachment_handbag     = torch.from_numpy(training_label[:,39])
    training_attachment_box         = torch.from_numpy(training_label[:,40])
    training_attachment_plasticbag  = torch.from_numpy(training_label[:,41])
    training_attachment_paperbag    = torch.from_numpy(training_label[:,42])
    training_attachment_handtrunk   = torch.from_numpy(training_label[:,43])
    training_attachment_other       = torch.from_numpy(training_label[:,44])
    training_action_calling         = torch.from_numpy(training_label[:,45])
    training_action_talking         = torch.from_numpy(training_label[:,46])
    training_action_gathering       = torch.from_numpy(training_label[:,47])
    training_action_holding         = torch.from_numpy(training_label[:,48])
    training_action_pushing         = torch.from_numpy(training_label[:,49])
    training_action_pulling         = torch.from_numpy(training_label[:,50])
    training_action_carryingbyarm   = torch.from_numpy(training_label[:,51])
    training_action_carryingbyhand  = torch.from_numpy(training_label[:,51])
    training_action_other           = torch.from_numpy(training_label[:,53])

    val_img           = torch.from_numpy(val_img)
    val_female                 = torch.from_numpy(val_label[:,0])
    val_ageless16              = torch.from_numpy(val_label[:,1])
    val_age17_30               = torch.from_numpy(val_label[:,2])
    val_age31_45               = torch.from_numpy(val_label[:,3])
    val_age46_60               = torch.from_numpy(val_label[:,4])
    val_bodyfat                = torch.from_numpy(val_label[:,5])
    val_bodynormal             = torch.from_numpy(val_label[:,6])
    val_bodythin               = torch.from_numpy(val_label[:,7])
    val_customer               = torch.from_numpy(val_label[:,8])
    val_employee               = torch.from_numpy(val_label[:,9])
    val_hs_baldhead            = torch.from_numpy(val_label[:,10])
    val_hs_longhair            = torch.from_numpy(val_label[:,11])
    val_hs_blackhair           = torch.from_numpy(val_label[:,12])
    val_hs_hat                 = torch.from_numpy(val_label[:,13])
    val_hs_glasses             = torch.from_numpy(val_label[:,14])
    val_ub_shirt               = torch.from_numpy(val_label[:,15])
    val_ub_sweater             = torch.from_numpy(val_label[:,16])
    val_ub_vest                = torch.from_numpy(val_label[:,17])
    val_ub_tshirt              = torch.from_numpy(val_label[:,18])
    val_ub_cotton              = torch.from_numpy(val_label[:,19])
    val_ub_jacket              = torch.from_numpy(val_label[:,20])
    val_ub_suitup              = torch.from_numpy(val_label[:,21])
    val_ub_tight               = torch.from_numpy(val_label[:,22])
    val_ub_shortsleeve         = torch.from_numpy(val_label[:,23])
    val_ub_others              = torch.from_numpy(val_label[:,24])
    val_lb_longtrousers        = torch.from_numpy(val_label[:,25])
    val_lb_skirt               = torch.from_numpy(val_label[:,26])
    val_lb_shortskirt          = torch.from_numpy(val_label[:,27])
    val_lb_dress               = torch.from_numpy(val_label[:,28])
    val_lb_jeans               = torch.from_numpy(val_label[:,29])
    val_lb_tighttrousers       = torch.from_numpy(val_label[:,30])
    val_shoes_leather          = torch.from_numpy(val_label[:,31])
    val_shoes_sports           = torch.from_numpy(val_label[:,32])
    val_shoes_boots            = torch.from_numpy(val_label[:,33])
    val_shoes_cloth            = torch.from_numpy(val_label[:,34])
    val_shoes_casual           = torch.from_numpy(val_label[:,35])
    val_shoes_others           = torch.from_numpy(val_label[:,36])
    val_attachment_backpack    = torch.from_numpy(val_label[:,37])
    val_attachment_shoulderbag = torch.from_numpy(val_label[:,38])
    val_attachment_handbag     = torch.from_numpy(val_label[:,39])
    val_attachment_box         = torch.from_numpy(val_label[:,40])
    val_attachment_plasticbag  = torch.from_numpy(val_label[:,41])
    val_attachment_paperbag    = torch.from_numpy(val_label[:,42])
    val_attachment_handtrunk   = torch.from_numpy(val_label[:,43])
    val_attachment_other       = torch.from_numpy(val_label[:,44])
    val_action_calling         = torch.from_numpy(val_label[:,45])
    val_action_talking         = torch.from_numpy(val_label[:,46])
    val_action_gathering       = torch.from_numpy(val_label[:,47])
    val_action_holding         = torch.from_numpy(val_label[:,48])
    val_action_pushing         = torch.from_numpy(val_label[:,49])
    val_action_pulling         = torch.from_numpy(val_label[:,50])
    val_action_carryingbyarm   = torch.from_numpy(val_label[:,51])
    val_action_carryingbyhand  = torch.from_numpy(val_label[:,51])
    val_action_other           = torch.from_numpy(val_label[:,53])
    
    return training_img,training_female,training_ageless16,training_age17_30,training_age31_45,training_age46_60,training_bodyfat,training_bodynormal,training_bodythin,training_customer,training_employee,training_hs_baldhead,training_hs_longhair,training_hs_blackhair,training_hs_hat,training_hs_glasses,training_ub_shirt,training_ub_sweater,training_ub_vest,training_ub_tshirt,training_ub_cotton,training_ub_jacket,training_ub_suitup,training_ub_tight,training_ub_shortsleeve,training_ub_others,training_lb_longtrousers,training_lb_skirt,training_lb_shortskirt,training_lb_dress,training_lb_jeans,training_lb_tighttrousers,training_shoes_leather,training_shoes_sports,training_shoes_boots,training_shoes_cloth,training_shoes_casual,training_shoes_others,training_attachment_backpack,training_attachment_shoulderbag,training_attachment_handbag,training_attachment_box,training_attachment_plasticbag,training_attachment_paperbag,training_attachment_handtrunk,training_attachment_other,training_action_calling,training_action_talking,training_action_gathering,training_action_holding,training_action_pushing,training_action_pulling,training_action_carryingbyarm,training_action_carryingbyhand,training_action_other,val_img,val_female,val_ageless16,val_age17_30,val_age31_45,val_age46_60,val_bodyfat,val_bodynormal,val_bodythin,val_customer,val_employee,val_hs_baldhead,val_hs_longhair,val_hs_blackhair,val_hs_hat,val_hs_glasses,val_ub_shirt,val_ub_sweater,val_ub_vest,val_ub_tshirt,val_ub_cotton,val_ub_jacket,val_ub_suitup,val_ub_tight,val_ub_shortsleeve,val_ub_others,val_lb_longtrousers,val_lb_skirt,val_lb_shortskirt,val_lb_dress,val_lb_jeans,val_lb_tighttrousers,val_shoes_leather,val_shoes_sports,val_shoes_boots,val_shoes_cloth,val_shoes_casual,val_shoes_others,val_attachment_backpack,val_attachment_shoulderbag,val_attachment_handbag,val_attachment_box,val_attachment_plasticbag,training_attachment_paperbag,val_attachment_handtrunk,val_attachment_other,val_action_calling,val_action_talking,val_action_gathering,val_action_holding,val_action_pushing,val_action_pulling,val_action_carryingbyarm,val_action_carryingbyhand,val_action_other

    