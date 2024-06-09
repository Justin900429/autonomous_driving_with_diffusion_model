import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetGuidance(nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()

    def forward(self, x, target):
        num_target = target.size(0)
        with torch.no_grad():
            dist_to_origin = torch.norm(target[0], dim=-1)
            dist_space = torch.norm(target[1:] - target[:-1], dim=-1).mean()
            num_to_insert = int(dist_to_origin / (dist_space + 1e-5))
            if num_to_insert > 0:
                new_points = []
                for i in range(num_to_insert):
                    new_points.append((i + 1) * target[0] / num_to_insert)
                target = torch.cat([torch.stack(new_points), target], dim=0)[
                    :num_target
                ]
        while target.dim() < x.dim():
            target = target.unsqueeze(0)

        dist_matrix = torch.sum(
            (x[..., :2].unsqueeze(2) - target.unsqueeze(1)) ** 2, dim=-1
        ).amin(
            dim=-1
        )  # (B, N)
        loss = dist_matrix[:, 1:].mean(dim=-1).sum()
        return loss
