"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Quick demo: create a tiny random bundle and sample once.

This validates wiring without training or external files.
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
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
    p = argparse.ArgumentParser(
        description="Write a tiny random bundle and sample once",
    )
    p.add_argument("--out", default="model_bundle_demo.pt")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device
    cfg = GPTConfig(
        vocab_size=256,
        block_size=64,
        d_model=64,
        n_head=4,
        n_layer=2,
        d_ff=128,
    )
    model = GPT(cfg).to(device).eval()
    bundle = {
        "config": cfg.__dict__,
        "model_state": model.state_dict(),
        "tokenizer": None,
    }
    torch.save(bundle, args.out)
    print("Wrote:", args.out)

    ids = torch.tensor(
        [[c for c in args.prompt.encode("utf-8")]],
        dtype=torch.long,
        device=device,
    )
    out = sample(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=(args.top_k or None),
        top_p=(args.top_p or None),
    )
    text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    print("Sample:\n", text)


if __name__ == "__main__":
    main()
