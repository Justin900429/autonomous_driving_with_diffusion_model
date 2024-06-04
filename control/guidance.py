import importlib
import os

import torch.nn as nn

DIR_NAME = os.path.dirname(__file__)


class GuidanceLoss(nn.Module):
    def __init__(self, loss_cls_list):
        super(GuidanceLoss, self).__init__()

        module = importlib.import_module(f"{DIR_NAME}.guidance_loss")
        self.loss_list = nn.ModuleList([])
        for loss_cls, loss_config in loss_cls_list:
            self.loss_list.append(getattr(module, loss_cls)(**loss_config))

    def forward(self, x, target):
        total_loss = 0
        for loss in self.loss_list:
            total_loss += loss(x, target)
        return total_loss
