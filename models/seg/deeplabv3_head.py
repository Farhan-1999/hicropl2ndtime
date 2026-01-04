# models/seg/deeplabv3_head.py
from __future__ import annotations

from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Standard DeepLabV3 ASPP module.
    """
    def __init__(self, in_ch: int, out_ch: int, atrous_rates: Sequence[int] = (6, 12, 18), dropout: float = 0.1):
        super().__init__()
        branches = []

        # 1x1
        branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )

        # 3x3 dilated
        for r in atrous_rates:
            branches.append(ASPPConv(in_ch, out_ch, dilation=int(r)))

        # image pooling
        branches.append(ASPPPooling(in_ch, out_ch))

        self.branches = nn.ModuleList(branches)

        proj_in = out_ch * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(proj_in, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3EmbeddingHead(nn.Module):
    """
    DeepLabV3-style decoder that outputs *dense pixel embeddings* (not class logits).

    Input:
      x: [B, in_ch, H', W']

    Output:
      emb: [B, embed_dim, H', W']  (upsampling to full res is done outside)
    """
    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        aspp_ch: int = 256,
        atrous_rates: Sequence[int] = (6, 12, 18),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aspp = ASPP(in_ch=in_ch, out_ch=aspp_ch, atrous_rates=atrous_rates, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.to_embed = nn.Conv2d(aspp_ch, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.fuse(x)
        x = self.to_embed(x)
        return x


class DeepLabV3PlusEmbeddingHead(nn.Module):
    """
    DeepLabV3+ style head (often higher mIoU): ASPP on deep feature + fuse low-level skip feature.

    Inputs:
      deep: [B, deep_ch, H', W']
      low:  [B, low_ch,  H_low, W_low]  (usually 2-4x higher res than H',W')

    Output:
      emb: [B, embed_dim, H_low, W_low]  (upsampling to full res is done outside)
    """
    def __init__(
        self,
        deep_ch: int,
        low_ch: int,
        embed_dim: int,
        aspp_ch: int = 256,
        low_proj_ch: int = 48,
        atrous_rates: Sequence[int] = (6, 12, 18),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aspp = ASPP(in_ch=deep_ch, out_ch=aspp_ch, atrous_rates=atrous_rates, dropout=dropout)

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_proj_ch),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + low_proj_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )

        self.to_embed = nn.Conv2d(aspp_ch, embed_dim, kernel_size=1, bias=True)

    def forward(self, deep: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        # ASPP on deep
        x = self.aspp(deep)  # [B, aspp_ch, H', W']

        # upsample to low-level resolution
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        # project low-level features
        lowp = self.low_proj(low)  # [B, low_proj_ch, H_low, W_low]

        # fuse and embed
        x = torch.cat([x, lowp], dim=1)
        x = self.fuse(x)
        x = self.to_embed(x)
        return x
