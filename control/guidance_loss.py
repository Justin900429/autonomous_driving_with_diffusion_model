import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()

    def forward(self, x, target):
        while target.dim() < x.dim():
            target = target.unsqueeze(0)

        loss_weight = F.softmin(torch.norm(target, dim=-1), dim=-1)
        dist_matrix = (
            torch.sum((x[..., :2].unsqueeze(1) - target.unsqueeze(2)) ** 2, dim=-1).amin(dim=-1)
            * loss_weight
        )  # (B, T)
        loss = dist_matrix.mean(dim=-1).sum()
        return loss
