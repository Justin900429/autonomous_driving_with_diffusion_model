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
        self.learning_rate = cfg.GUIDANCE.LR
        
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss in self.loss_list:
            total_loss += loss(x, target)
        return total_loss

    def forward(self, x_init: torch.Tensor, target: torch.Tensor, guidance_threshold: Union[float, torch.Tensor] = None)-> torch.Tensor:
        x_guidance = x_init.clone().detach()
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()
        guidance_threshold = guidance_threshold.to(x_guidance.device) if guidance_threshold is not None else None
        opt = torch.optim.Adam([x_guidance], lr=self.learning_rate)
        
        for _ in range(self.guidance_step):
            with torch.enable_grad():
                loss = self.compute_loss(x_guidance, target)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if guidance_threshold is not None:
                with torch.no_grad():
                    x_delta = x_guidance - x_init
                    x_delta = torch.clamp(x_delta, -guidance_threshold, guidance_threshold)
                    x_guidance.data = x_init + x_delta
        x_guidance.requires_grad_(False)
        return x_guidance