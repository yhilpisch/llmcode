from __future__ import annotations

"""Sample text from a trained checkpoint (Chapter 11 companion CLI).

This script reconstructs the GPT model saved by Chapter 10's trainer and
generates a short continuation from a prompt. It is byte-level by default so
you can sample without extra tokenizer files.

Examples
--------
(.venv) $ python code/sample_from_checkpoint.py \
  --ckpt checkpoints/ch10_gpt.pt --prompt "Philosophy is" \
  --max-new-tokens 120 --temperature 0.9 --top-p 0.95

Notes
-----
- If you trained with a custom tokenizer from Chapter 6, pass a prompt that is
  compatible with your vocabulary or adapt this script to encode/decode with
  that tokenizer.
"""

import argparse
import sys
from typing import Optional

import torch

# Make "code/" importable when running as a script
from pathlib import Path
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


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Sample from a GPT checkpoint")
    p.add_argument("--ckpt", default="checkpoints/ch10_gpt.pt", help="path to .pt")
    p.add_argument("--prompt", default="Hello", help="Prompt text")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", default="auto", choices=["auto", "byte", "char", "word"],
                   help="tokenization level for prompt/decoding")
    p.add_argument("--ref-text", nargs='*', default=[],
                   help="text file(s) used to rebuild tokenizer for char/word levels")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Determine tokenization/decoding strategy
    level = args.level
    if level == "auto":
        # Heuristic: byte-level models typically have vocab_size==256
        level = "byte" if cfg.vocab_size == 256 else "char"

    if level == "byte":
        prompt_bytes = args.prompt.encode("utf-8")
        input_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
        out = sample(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=(args.top_k if args.top_k > 0 else None),
            top_p=(args.top_p if args.top_p > 0 else None),
        )
        text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
        print(text)
    else:
        # Prefer tokenizer embedded in checkpoint; otherwise rebuild from refs.
        from ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore
        tok = None
        if "tokenizer" in ckpt:
            meta = ckpt["tokenizer"]
            if meta.get("level") == level and meta.get("id_to_token"):
                id_to_token = list(meta["id_to_token"])  # ensure list
                token_to_id = {t: i for i, t in enumerate(id_to_token)}
                pad_id = int(meta.get("pad_id", 0))
                unk_id = int(meta.get("unk_id", 1))
                vocab = Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad=pad_id, unk=unk_id)
                tok = SimpleTokenizer(vocab=vocab, level=level)
        if tok is None:
            if not args.ref_text:
                print("ERROR: provide --ref-text files to rebuild tokenizer for level=char/word.")
                sys.exit(2)
            ref = "\n".join(Path(p).read_text(encoding="utf-8") for p in args.ref_text)
            tokens = SimpleTokenizer._split(ref, level)
            vocab = Vocab.build(tokens)
            tok = SimpleTokenizer(vocab=vocab, level=level)
            if len(vocab) != cfg.vocab_size:
                print(f"WARNING: tokenizer vocab {len(vocab)} != model vocab {cfg.vocab_size}; decoding may be off.")
        input_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
        out = sample(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=(args.top_k if args.top_k > 0 else None),
            top_p=(args.top_p if args.top_p > 0 else None),
        )
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
