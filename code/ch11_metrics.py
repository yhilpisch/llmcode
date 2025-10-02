from __future__ import annotations

"""Simple evaluation helpers for Chapter 11.

Perplexity is derived from average cross-entropy on a held-out set:
    PPL = exp(H)
We compute mean loss over a DataLoader of (x, y) pairs.
"""

import math
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, loader) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, targets=y)
        if loss is None:
            # fallback: compute CE manually
            B, T, V = logits.shape
            lf = logits.reshape(B * T, V)
            yf = y.reshape(B * T)
            loss = F.cross_entropy(lf, yf)
        n = y.numel()
        total_loss += float(loss.detach().item()) * n
        total_tokens += int(n)
    H = total_loss / max(1, total_tokens)
    return H, math.exp(H)


__all__ = ["perplexity"]

