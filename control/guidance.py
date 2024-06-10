import importlib
import os
from typing import Union

import torch
import torch.nn as nn

DIR_NAME = os.path.basename(os.path.dirname(__file__))


def convert(loss_config):
    it = iter(loss_config)
    res_dct = dict(zip(it, it))
    return res_dct


class GuidanceLoss(nn.Module):
    def __init__(self, cfg):
        super(GuidanceLoss, self).__init__()

        module = importlib.import_module(f"{DIR_NAME}.guidance_loss")
        self.loss_list = nn.ModuleList([])
        for loss_cls, loss_config in cfg.GUIDANCE.LOSS_LIST:
            self.loss_list.append(getattr(module, loss_cls)(**convert(loss_config)))

        self.guidance_step = cfg.GUIDANCE.STEP
        self.scale = cfg.GUIDANCE.CLASSIFIER_SCALE

    def compute_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss in self.loss_list:
            total_loss += loss(x, target)
        return total_loss

    def forward(
        self,
        x_init: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        grad_scale: Union[float, torch.Tensor] = None,
    ) -> torch.Tensor:
        x_guidance = x_init
        for _ in range(self.guidance_step):
            with torch.enable_grad():
                if not x_guidance.requires_grad:
                    x_guidance.requires_grad_()
                loss = self.compute_loss(x_guidance, target)
                action_grad = torch.autograd.grad(
                    [loss],
                    [action],
                    retain_graph=True,
                )[0]
                state_grad = torch.autograd.grad(
                    [loss],
                    [x_guidance],
                )[
                    0
                ][..., :-3]
                grad = torch.cat([state_grad, action_grad], dim=-1)
            if grad_scale is not None:
                grad *= grad_scale
            x_guidance = x_guidance.detach()
            x_guidance = x_guidance - self.scale * grad
        x_guidance.requires_grad_(False)
        return x_guidance
