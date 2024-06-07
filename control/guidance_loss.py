import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self, target_index: int = 4):
        super(TargetGuidance, self).__init__()
        self.target_index = target_index

    def forward(self, x, target):
        while target.dim() < x.dim() - 1:
            target = target.unsqueeze(0)
        return F.mse_loss(x[:, self.target_index, :2], target).mean()
