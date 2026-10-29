"""U-Net noise predictor: ResBlocks (GroupNorm + SiLU) with time/class conditioning,
strided-conv down-sampling, nearest-neighbour up-sampling, skip connections, and
self-attention at low resolution."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps, shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return F.pad(emb, (0, dim % 2))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)  # blocks start as identity: stabilises early training
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]  # inject (time [+ class]) embedding
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """epsilon_theta(x_t, t [, y]).  Image side must be divisible by 2**(len(mults)-1).

    With `num_classes`, a learned class embedding (plus one extra "null" index used for
    classifier-free guidance) is added to the time embedding.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        mults: tuple[int, ...] = (1, 2, 2),
        num_res: int = 2,
        attn_below: int = 8,
        image_size: int = 28,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base, self.num_classes = base, num_classes
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim) if num_classes else None
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        chans, res = [base], image_size
        ch = base
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(ch, out, emb_dim, dropout=dropout)])
                if res < attn_below:
                    blk.append(SelfAttention(out))
                self.down.append(blk)
                ch = out
                chans.append(ch)
            if i < len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                chans.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout=dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout=dropout)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for j in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), out, emb_dim, dropout=dropout)])
                if res < attn_below:
                    blk.append(SelfAttention(out))
                ch = out
                if i > 0 and j == num_res:
                    blk.append(Upsample(ch))
                    res *= 2
                self.up.append(blk)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base))
        if self.class_emb is not None:
            if y is None:  # unconditional: the reserved null class
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)

        h = self.conv_in(x)
        skips = [h]
        for blk in self.down:
            for layer in blk:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for blk in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in blk:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))
