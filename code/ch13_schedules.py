"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Learning-rate schedule helpers (Chapter 13).

Includes a warmup+cosine decay schedule implemented via PyTorch's LambdaLR.
The schedule scales the base LR by a factor in [min_ratio, 1].
"""

from __future__ import annotations


import math
from typing import Optional

import torch


def warmup_cosine_lambda(
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
):
    """Return a lambda(step) for LambdaLR implementing warmup+cosine decay.

    - Warmup: linearly scale 0 -> 1 over warmup_steps.
    - Cosine: decay from 1 -> min_ratio over the remaining steps.
    """

    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    assert 0.0 < min_ratio <= 1.0

    def lr_lambda(step: int) -> float:
        s = step + 1
        if warmup_steps > 0 and s <= warmup_steps:
            return s / float(warmup_steps)
        # cosine from warmup_steps..total_steps
        t = min(max(s - warmup_steps, 0), max(total_steps - warmup_steps, 1))
        frac = t / float(max(total_steps - warmup_steps, 1))
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        return min_ratio + (1 - min_ratio) * cos

    return lr_lambda


def warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a LambdaLR with warmup+cosine schedule."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(warmup_steps, total_steps, min_ratio)
    )


__all__ = ["warmup_cosine_lr", "warmup_cosine_lambda"]

