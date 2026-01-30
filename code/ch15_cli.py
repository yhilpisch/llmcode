"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Minimal sampling CLI over an exported bundle (Chapter 15).

Usage:
  python code/ch15_cli.py --bundle model_bundle.pt --prompt "Hello"
"""

from __future__ import annotations


import argparse
import sys
from pathlib import Path

import torch

# Import code/ modules directly when run as a script
sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch11_sampling import sample  # type: ignore
from ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_tokenizer(meta: dict | None):
    if not meta:
        return None
    try:
        id_to_token = list(meta["id_to_token"])  # ensure list
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        pad_id = int(meta.get("pad_id", 0))
        unk_id = int(meta.get("unk_id", 1))
        vocab = Vocab(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad=pad_id,
            unk=unk_id,
        )
        return SimpleTokenizer(vocab=vocab, level=meta.get("level", "char"))
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from a GPT bundle")
    p.add_argument("--bundle", required=True, help="bundle .pt from ch15_export")
    p.add_argument("--prompt", required=True, help="prompt string")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device
    print({"device": device, "seed": args.seed})
    b = torch.load(args.bundle, map_location=device)
    cfg = GPTConfig(**b["config"])  # type: ignore
    model = GPT(cfg).to(device)
    model.load_state_dict(b["model_state"])  # type: ignore
    model.eval()

    tok = build_tokenizer(b.get("tokenizer"))
    if tok is None:  # fall back to byte-level
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
        print(bytes(out[0].tolist()).decode("utf-8", errors="ignore"))
    else:
        ids = torch.tensor(
            [tok.encode(args.prompt)],
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
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
