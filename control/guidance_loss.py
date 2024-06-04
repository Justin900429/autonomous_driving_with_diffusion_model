import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()

    def forward(self, x, target):
        while target.dim() < x.dim():
            target = target.unsqueeze(0)
        return F.mse_loss(x[:, -1, :2], target).mean()
