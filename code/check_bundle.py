"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Validate a model bundle by loading it and sampling once.
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
import torch

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


def main() -> None:
    p = argparse.ArgumentParser(description="Bundle smoke-test: load and sample")
    p.add_argument("--bundle", required=True, help="path to model_bundle.pt")
    p.add_argument("--prompt", default="Hello", help="prompt string")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="auto", help="cpu|cuda|mps|auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Make sampling deterministic and pick a device
    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device
    print({"device": device, "seed": args.seed})

    # Load bundle, restore model and optional tokenizer
    b = torch.load(args.bundle, map_location=device)
    cfg = GPTConfig(**b["config"])  # type: ignore
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(b["model_state"])  # type: ignore
    meta = b.get("tokenizer")
    tok = None
    if meta and meta.get("id_to_token"):
        id_to_token = list(meta["id_to_token"])  # ensure list
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        vocab = Vocab(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad=int(meta.get("pad_id", 0)),
            unk=int(meta.get("unk_id", 1)),
        )
        tok = SimpleTokenizer(vocab=vocab, level=meta.get("level", "char"))

    if tok is None:
        # Fallback to byte-level prompt if no tokenizer metadata exists
        ids = torch.tensor(
            [[c for c in args.prompt.encode("utf-8")]],
            dtype=torch.long,
            device=device,
        )
        out = sample(
            model, ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=(args.top_k or None),
            top_p=(args.top_p or None),
        )
        text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    else:
        ids = torch.tensor(
            [tok.encode(args.prompt)], dtype=torch.long, device=device
        )
        out = sample(
            model, ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=(args.top_k or None),
            top_p=(args.top_p or None),
        )
        text = tok.decode(out[0].tolist())
    print("OK — model loaded and sampled.\n", text)


if __name__ == "__main__":
    main()
