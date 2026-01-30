"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Measure forward-only tokens/sec for a tiny GPT.

Usage:
  python code/bench_forward.py --device auto --batch 8 --block 128 --vocab 256
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
import time
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()

    # Resolve device string lazily for portable benchmarking
    device = auto_device() if args.device == "auto" else args.device
    cfg = GPTConfig(vocab_size=args.vocab, block_size=args.block)
    # Keep the model tiny to highlight kernel overheads
    model = GPT(cfg).to(device).eval()
    # Synthetic token batch to avoid disk access
    x = torch.randint(
        0, cfg.vocab_size, (args.batch, cfg.block_size), device=device
    )

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    tok = 0
    for _ in range(args.steps):
        with torch.no_grad():
            model(x)
        tok += args.batch * args.block
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print({"device": device, "tokens_per_sec": round(tok / dt)})


if __name__ == "__main__":
    main()
