from __future__ import annotations

"""LoRA: Low-rank adapters for Linear layers (teaching version).

This module provides a small, readable `LoRALinear` that adds a trainable
low-rank delta to a frozen base weight:

    y = x @ W^T + scale * x @ (B @ A)^T

where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, and `scale = alpha / r`.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        r: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
    ) -> None:
        """Create a Linear with LoRA adapters.

        - d_in, d_out: base dimensions
        - r: adapter rank (small)
        - alpha: scaling factor (effective scale = alpha / r)
        - bias: include bias term on the base layer
        """
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)
        # LoRA adapters (A: r×d_in, B: d_out×r)
        if self.r > 0:
            self.A = nn.Linear(d_in, self.r, bias=False)
            self.B = nn.Linear(self.r, d_out, bias=False)
            # Init: A small, B zero so start as identity (delta≈0)
            nn.init.kaiming_uniform_(self.A.weight, a=2**0.5)
            nn.init.zeros_(self.B.weight)
            # Freeze base
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            self.A = None
            self.B = None
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and not self.merged:
            y = y + self.scale * self.B(self.A(x))
        return y

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the LoRA delta into the base weight for inference.

        After merging, adapters are disabled and the module acts like a
        standard Linear layer with updated weights.
        """
        if self.r == 0 or self.merged:
            self.merged = True
            return
        # W' = W + scale * (B @ A)
        delta = (self.B.weight @ self.A.weight) * self.scale
        self.base.weight += delta
        self.merged = True


__all__ = ["LoRALinear"]

