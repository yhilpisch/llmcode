"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Tiny sampling helpers (preview for Chapter 11).

Functions here keep dependencies minimal and work directly with the GPT model
from Chapter 9. They operate on integer token ids and return extended ids.
"""

from __future__ import annotations


from typing import Optional

import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[:, [-1]]
    return torch.where(logits < thresh, torch.tensor(-1e9, device=logits.device), logits)


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0 or p >= 1:
        return logits
    # sort descending and keep smallest set whose cumulative prob >= p
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > p
    # always keep the first token
    mask[..., 0] = False
    filtered = sorted_logits.masked_fill(mask, -1e9)
    # unsort back to original order
    unsorted = torch.empty_like(filtered).scatter_(1, sorted_idx, filtered)
    return unsorted


@torch.no_grad()
def sample(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively.

    - temperature: 0 → greedy (argmax); >0 → softmax sampling
    - top_k: keep only the top‑k logits at each step (optional)
    - eos_id: if set, stop when generated
    """
    model.eval()
    x = input_ids
    device = next(model.parameters()).device
    x = x.to(device)

    for _ in range(max_new_tokens):
        # Forward pass on the last block_size tokens
        T = x.size(1)
        block_size = getattr(model.cfg, "block_size", T)
        x_cond = x[:, -block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]  # last position

        if temperature <= 0:
            # Greedy
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                logits = _top_k_filter(logits, top_k)
            if top_p is not None:
                logits = _top_p_filter(logits, float(top_p))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)
        if eos_id is not None and int(next_id[0, 0].item()) == int(eos_id):
            break
    return x


__all__ = ["sample", "_top_k_filter", "_top_p_filter"]
