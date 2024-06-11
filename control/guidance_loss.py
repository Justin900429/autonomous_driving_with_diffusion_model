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
        dist_matrix = torch.sum((x[..., :2].unsqueeze(1) - target.unsqueeze(2)) ** 2, dim=-1)
        target_to_agent = torch.norm(target - x[:, 0, :2], dim=-1)
        final_to_agent = torch.norm(x[:, -1, :2] - x[:, 0, :2], dim=-1)
        if final_to_agent < target_to_agent:
            choose_target = 0  # Dummy point to prevent erratic update
        else:
            choose_target = dist_matrix.argmin(dim=-1)
        return (dist_matrix[:, :, choose_target] * loss_weight).mean(dim=-1).sum()
