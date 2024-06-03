import torch


class TargetGuidance(torch.nn.Module):
    def __init__(self):
        super(TargetGuidance, self).__init__()
        
    def forward(self, x, target):
        return torch.nn.functional.mse_loss(x[-1], target).mean()
