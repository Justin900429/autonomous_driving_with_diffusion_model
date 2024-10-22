import os

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from misc.constant import GuidanceType

from .helpers import (
    Conv1dBlock,
    Downsample1d,
    LinearAttention,
    PreNorm,
    Residual,
    SinusoidalPosEmb,
    TrajPredict,
    Upsample1d,
)
from .resnet import resnet34


class ResidualTemporalMapBlockConcat(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalMapUnet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim=2,
        attention=False,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        diffuser_building_block="concat",
        use_cond=GuidanceType.NO_GUIDANCE,
    ):
        super().__init__()

        if diffuser_building_block == "concat":
            ResidualTemporalMapBlock = ResidualTemporalMapBlockConcat
        else:
            raise NotImplementedError

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if int(os.environ.get("LOCAL_RANK", "-1")) <= 0:
            print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.perception = resnet34(pretrained=True)
        self.perception.fc = nn.Linear(self.perception.fc.in_features, time_dim)

        self.use_cond = use_cond
        if use_cond == GuidanceType.FREE_GUIDANCE:
            self.cond_mlp = nn.Sequential(
                nn.Linear(2, time_dim),
                nn.Mish(),
                nn.Linear(time_dim, time_dim),
            )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = time_dim + time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalMapBlock(
                            dim_in,
                            dim_out,
                            embed_dim=cond_dim,
                        ),
                        ResidualTemporalMapBlock(
                            dim_out,
                            dim_out,
                            embed_dim=cond_dim,
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalMapBlock(
            mid_dim,
            mid_dim,
            embed_dim=cond_dim,
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalMapBlock(
            mid_dim,
            mid_dim,
            embed_dim=cond_dim,
        )

        final_up_dim = None
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalMapBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=cond_dim,
                        ),
                        ResidualTemporalMapBlock(
                            dim_in,
                            dim_in,
                            embed_dim=cond_dim,
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            final_up_dim = dim_in

            if not is_last:
                horizon = horizon * 2

        if use_cond == GuidanceType.CLASSIFIER_GUIDANCE:
            state_dim = transition_dim - 3
            self.act_conv = nn.Sequential(
                Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
                nn.Conv1d(final_up_dim, 3, 1),
            )
            self.state_pred = TrajPredict(
                in_dim=3, out_dim=state_dim, pred_len=horizon - 1, hidden_dim=64, num_layers=2
            )
        else:
            self.final_conv = nn.Sequential(
                Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
                nn.Conv1d(final_up_dim, transition_dim, 1),
            )
        self.magic_num = 23.315

    def forward(self, x, img, time, cond=None, return_action_and_time_only=False):
        """
        x : [ B, T, D ]
        img : [ B, C, H, W]
        cond: None or [B, 2]
        """
        img_feature = self.perception(img)
        x = einops.rearrange(x, "b h t -> b t h")
        time_embed = self.time_mlp(time)
        if self.use_cond == GuidanceType.FREE_GUIDANCE:
            cond = cond if cond is not None else torch.zeros((x.shape[0], 2), device=x.device)
            if time_embed.shape[0] != cond.shape[0]:
                time_embed = time_embed.repeat(cond.shape[0] // time_embed.shape[0], 1)
            if img_feature.shape[0] != cond.shape[0]:
                img_feature = img_feature.repeat(cond.shape[0] // img_feature.shape[0], 1)
            time_embed += self.cond_mlp(cond)
        cond_input = torch.cat([time_embed, img_feature], dim=-1)

        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond_input)
            x = resnet2(x, cond_input)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, cond_input)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond_input)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond_input)
            x = resnet2(x, cond_input)
            x = attn(x)
            x = upsample(x)

        if self.use_cond == GuidanceType.CLASSIFIER_GUIDANCE:
            action = self.act_conv(x)
            action = einops.rearrange(action, "b t h -> b h t")
            if return_action_and_time_only:
                return action, time_embed
            state = self.state_pred(action.detach()[:, :-1], time_embed)
            # Setup a dummy zero in the intial state
            state = torch.cat([torch.zeros_like(state[:, :1]), state], dim=1)
            x = torch.cat([state, action], dim=-1)
        else:
            x = self.final_conv(x)
            x = einops.rearrange(x, "b t h -> b h t")
        return x


def build_model(cfg) -> TemporalMapUnet:
    model = TemporalMapUnet(
        horizon=cfg.MODEL.HORIZON,
        transition_dim=cfg.MODEL.TRANSITION_DIM,
        attention=cfg.MODEL.USE_ATTN,
        dim=cfg.MODEL.DIM,
        dim_mults=cfg.MODEL.DIM_MULTS,
        diffuser_building_block=cfg.MODEL.DIFFUSER_BUILDING_BLOCK,
        use_cond=GuidanceType[cfg.TRAIN.USE_COND],
    )
    return model
