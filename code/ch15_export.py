from __future__ import annotations

"""Export a clean model bundle with config, weights, and tokenizer metadata.

Usage:
  python code/ch15_export.py --ckpt checkpoints/ch13_gpt_best.pt --out model_bundle.pt
"""

import argparse
from pathlib import Path
import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Export GPT bundle")
    p.add_argument("--ckpt", required=True, help="input checkpoint .pt")
    p.add_argument("--out", required=True, help="output bundle .pt")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    bundle = {
        "config": ckpt.get("config"),
        "model_state": ckpt.get("model_state"),
        "tokenizer": ckpt.get("tokenizer"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()

