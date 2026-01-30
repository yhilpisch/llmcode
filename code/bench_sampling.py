"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Measure sampling tokens/sec for a tiny GPT.

Usage:
  python code/bench_sampling.py --device auto --max-new-tokens 200
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
import time
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch11_sampling import sample  # type: ignore


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
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    args = p.parse_args()

    # Choose device automatically unless explicitly set
    device = auto_device() if args.device == "auto" else args.device
    cfg = GPTConfig(vocab_size=args.vocab, block_size=args.block)
    # Construct a tiny model and prompt to isolate sampling speed
    model = GPT(cfg).to(device).eval()
    prompt = torch.randint(
        0, cfg.vocab_size, (1, min(8, args.block)), device=device
    )

    t0 = time.time()
    out = sample(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=(args.top_k or None),
        top_p=(args.top_p or None),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out.size(1) - prompt.size(1)
    print(
        {
            "device": device,
            "gen_tokens": int(gen),
            "tokens_per_sec": round(gen / dt),
        }
    )

if __name__ == '__main__':
    main()
