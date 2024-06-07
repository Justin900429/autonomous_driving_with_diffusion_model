import importlib
import os
from typing import Union

import torch
import torch.nn as nn

DIR_NAME = os.path.basename(os.path.dirname(__file__))


class GuidanceLoss(nn.Module):
    def __init__(self, cfg):
        super(GuidanceLoss, self).__init__()

        module = importlib.import_module(f"{DIR_NAME}.guidance_loss")
        self.loss_list = nn.ModuleList([])
        for loss_cls, loss_config in cfg.GUIDANCE.LOSS_LIST:
            self.loss_list.append(getattr(module, loss_cls)(**loss_config))

        self.guidance_step = cfg.GUIDANCE.STEP
        self.scale = cfg.GUIDANCE.SCALE

    def compute_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss in self.loss_list:
            total_loss += loss(x, target)
        return total_loss

    def forward(
        self,
        x_init: torch.Tensor,
        target: torch.Tensor,
        grad_scale: Union[float, torch.Tensor] = None,
    ) -> torch.Tensor:
        x_guidance = x_init.clone().detach()
        for _ in range(self.guidance_step):
            with torch.enable_grad():
                if not x_guidance.requires_grad:
                    x_guidance.requires_grad_()
                loss = self.compute_loss(x_guidance, target)
                grad = torch.autograd.grad([loss], [x_guidance])[0]
            if grad_scale is not None:
                grad *= grad_scale
            x_guidance = x_guidance.detach()
            x_guidance = x_guidance - 50 * grad
        x_guidance.requires_grad_(False)
        return x_guidance
