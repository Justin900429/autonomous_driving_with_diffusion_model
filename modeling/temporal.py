from collections import deque

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .helpers import (
    Conv1dBlock,
    Downsample1d,
    LinearAttention,
    PreNorm,
    Residual,
    SinusoidalPosEmb,
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
    ):
        super().__init__()

        if diffuser_building_block == "concat":
            ResidualTemporalMapBlock = ResidualTemporalMapBlockConcat
        else:
            raise NotImplementedError

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.perception = resnet34(pretrained=True)
        self.perception.fc = nn.Linear(self.perception.fc.in_features, time_dim)

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

        self.final_conv = nn.Sequential(
            Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
            nn.Conv1d(final_up_dim, transition_dim, 1),
        )
        self.magic_num = 23.315

    def forward(self, x, img, time):
        """
        x : [ B, T, D ]
        cond_feat : [ B, C, H, W]
        """
        img_feature = self.perception(img)
        x = einops.rearrange(x, "b h t -> b t h")
        t = self.time_mlp(time)
        t = torch.cat([t, img_feature], dim=-1)

        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

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
    )
    return model
