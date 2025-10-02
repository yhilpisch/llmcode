from __future__ import annotations

"""Self-test: verify GPT forward shapes and mask broadcasting.

Runs a tiny forward pass and asserts expected tensor ranks/shapes for a
minimal config. Use this as a quick wiring check during refactors.

Usage:
  python code/gpt_shapes_selftest.py
"""

from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore


def main() -> None:
    cfg = GPTConfig(vocab_size=256, block_size=8, d_model=64, n_head=4, n_layer=2, d_ff=128)
    model = GPT(cfg)
    B, T = 2, 8
    x = torch.randint(0, cfg.vocab_size, (B, T))
    pad_id = None
    logits, loss = model(x, targets=x, pad_id=pad_id)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    assert loss is not None and loss.ndim == 0
    # Check causal mask shape indirectly via attention path: run shorter T
    T2 = 5
    x2 = torch.randint(0, cfg.vocab_size, (B, T2))
    logits2, _ = model(x2)
    assert logits2.shape[:2] == (B, T2)
    print("OK — GPT shapes and mask broadcasting look good.")


if __name__ == "__main__":
    main()

