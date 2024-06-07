import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()

    def forward(self, x, target):
        while target.dim() < x.dim():
            target = target.unsqueeze(0)
        with torch.no_grad():
            dist = torch.norm(x[..., :2] - target, dim=-1)
            loss_weighting = F.softmin(dist, dim=-1)
            loss_weighting[:, 0] = 0
        return (loss_weighting * torch.sum((x[..., :2] - target)**2, dim=-1)).mean(dim=-1).sum()
        # return torch.sum((x[:, 1:, :2] - target)**2, dim=-1).mean(dim=-1).sum()
