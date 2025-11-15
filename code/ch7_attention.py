"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

"""

from __future__ import annotations

import torch
from torch import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Single-head scaled dot-product attention.

    Args:
        q,k,v: [B, T, D]
        mask: optional [B, T, T] with 1 for allowed positions, 0 for masked
    Returns:
        [B, T, D]
    """
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # [B, T, T]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(scores, dim=-1)                 # [B, T, T]
    return w @ v                                      # [B, T, D]


def causal_mask(batch: int, time: int, device: torch.device | None = None) -> Tensor:
    base = torch.tril(torch.ones(time, time, device=device))  # [T, T]
    return base.unsqueeze(0).expand(batch, -1, -1)            # [B, T, T]


__all__ = ["scaled_dot_product_attention", "causal_mask"]

