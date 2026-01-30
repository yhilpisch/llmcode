"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(
    T: int,
    d_model: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [T, d_model] sinusoidal position encodings (sin/cos pairs).

    - T: sequence length (time steps)
    - d_model: embedding dimension (even recommended)
    """
    # positions [T,1] and index grid [1,D]
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(d_model, device=device).float()[None, :]
    # frequency grid [T,D]
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    enc = torch.zeros(T, d_model, device=device)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])  # even dims = sin
    enc[:, 1::2] = torch.cos(angle[:, 1::2])  # odd  dims = cos
    return enc


class MultiHeadAttention(nn.Module):
    """Multi‑head self‑attention (single module, H heads).

    d_model = H * Dh, where Dh is per‑head dim. We project to Q,K,V, split into
    heads, apply scaled dot‑product attention per head, then concat and project out.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads  # number of heads
        self.d = d_model // num_heads  # per‑head dim
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # shared proj
        self.out = nn.Linear(d_model, d_model, bias=False)  # output proj
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, Dm = x.shape  # batch, time, model dim
        qkv = self.qkv(x)  # [B, T, 3*Dm]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, Dm]

        # Split heads: [B,T,Dm] -> [B,H,T,Dh],
        # then put heads dimension before time.
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.h, self.d).transpose(1, 2)

        q, k, v = map(split, (q, k, v))  # [B, H, T, Dh]

        # Build [B, H, T, T] boolean mask (True = disallowed)
        mask_bool = None
        if mask is not None:
            if mask.dim() == 2:
                mask_bool = (
                    (mask == 0)
                    .bool()[None, None, :, :]
                    .expand(B, self.h, T, T)
                )
            elif mask.dim() == 3:
                mask_bool = (
                    (mask == 0)
                    .bool()
                    .unsqueeze(1)
                    .expand(B, self.h, T, T)
                )
            elif mask.dim() == 4:
                if mask.size(1) == 1:
                    mask_bool = (
                        (mask == 0).bool().expand(B, self.h, T, T)
                    )
                else:
                    mask_bool = (mask == 0).bool()

        # Manual scaled dot‑product attention for portability (MPS-safe)
        Dh = self.d
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)  # [B,H,T,T]
        if mask_bool is not None:
            scores = scores.masked_fill(mask_bool, float(-1e9))
        w = torch.softmax(scores, dim=-1)
        attn = w @ v  # [B,H,T,Dh]
        attn = self.drop(attn)

        # Concatenate heads back: [B, H, T, Dh] -> [B, T, Dm]
        y = (
            attn.transpose(1, 2)
            .contiguous()
            .view(B, T, Dm)
        )
        return self.out(y)


class FeedForward(nn.Module):
    """Position‑wise MLP with GELU and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # expand
            nn.GELU(),  # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),  # project back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Residual(nn.Module):
    """Pre‑norm residual wrapper: x + sublayer(LN(x))."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x + sublayer(self.norm(x), *args, **kwargs)


class TransformerBlock(nn.Module):
    """One pre‑norm transformer block: MHA + FFN with residuals."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.res1 = Residual(d_model)
        self.res2 = Residual(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.res1(x, self.mha, mask)  # attend + residual
        x = self.res2(x, self.ffn)  # think (FFN) + residual
        return x


__all__ = [
    "sinusoidal_positions",
    "MultiHeadAttention",
    "FeedForward",
    "Residual",
    "TransformerBlock",
]
