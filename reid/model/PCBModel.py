import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class PCBModel(nn.Module):
  def __init__(
      self,
      last_conv_stride=1,
      last_conv_dilation=1,
      num_stripes=6,
      local_conv_out_channels=2048,
      num_classes=1295
  ):
    super(PCBModel, self).__init__()

    self.base = resnet50(pretrained=True)
    
    self.global_conv = nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                     nn.BatchNorm2d(local_conv_out_channels),
                                     nn.ReLU(inplace=True)
                                    )
    self.global_fc = nn.Linear(local_conv_out_channels, num_classes)
    init.normal_(self.global_fc.weight, std=0.001)
    init.constant_(self.global_fc.bias, 0)
    
    self.num_stripes = num_stripes

    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes):
      self.local_conv_list.append(nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                                nn.BatchNorm2d(local_conv_out_channels),
                                                nn.ReLU(inplace=True)
      ))

    if num_classes > 0:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes):      
        fc = nn.Linear(local_conv_out_channels, num_classes)
        init.normal_(fc.weight, std=0.001)
        init.constant_(fc.bias, 0)
        self.fc_list.append(fc)


  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    
    assert feat.size(2) % self.num_stripes == 0
    stripe_h = int(feat.size(2) / self.num_stripes)
    local_feat_list = []
    logits_list = []
    for i in range(self.num_stripes):
        # shape [N, C, 1, 1]
        local_feat = F.avg_pool2d(
        feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
        (stripe_h, feat.size(-1)))

        # shape [N, c, 1, 1]
        local_feat = self.local_conv_list[i](local_feat)

        # shape [N, c]
        local_feat = local_feat.view(local_feat.size(0), -1)
        local_feat_list.append(local_feat)

        logits_list.append(self.fc_list[i](local_feat))
    
    global_feat = self.global_conv(F.adaptive_avg_pool2d(feat, 1))
    global_feat = global_feat.view(global_feat.size(0), -1)
    global_logits = self.global_fc(global_feat)
        
    local_feat_list.append(global_feat)
    logits_list.append(global_logits)
    
    return local_feat_list, logits_list
