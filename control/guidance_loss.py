import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()

    def forward(self, x, target):
        num_target = target.size(0)
        # with torch.no_grad():
        #     dist_to_origin = torch.norm(target[0], dim=-1)
        #     dist_space = torch.norm(target[1:] - target[:-1], dim=-1).mean()
        #     num_to_insert = int(dist_to_origin / (dist_space + 1e-5))
        #     if num_to_insert > 0:
        #         new_points = []
        #         for i in range(num_to_insert):
        #             new_points.append((i + 1) * target[0] / num_to_insert)
        #         target = torch.cat([torch.stack(new_points), target], dim=0)[:num_target]
        while target.dim() < x.dim():
            target = target.unsqueeze(0)

        loss_weight = F.softmin(torch.norm(target, dim=-1), dim=-1)
        dist_matrix = (
            torch.sum((x[..., :2].unsqueeze(1) - target.unsqueeze(2)) ** 2, dim=-1).amin(dim=-1)
            * loss_weight
        )  # (B, T)
        loss = dist_matrix.mean(dim=-1).sum()
        return loss
