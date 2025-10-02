from __future__ import annotations

"""Chapter 10: a compact training script for the GPT model.

This keeps options small and readable. It supports either a byte-level build
of token ids or the SimpleTokenizer from Chapter 6 if available.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make "code/" directory importable when running as script
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch10_data import (  # type: ignore
    LMSequenceDataset,
    build_ids_byte_level,
    build_ids_with_tokenizer,
    load_texts,
)


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny GPT (Chapter 10)")
    p.add_argument("--data", nargs="*", help="text file(s) to train on")
    p.add_argument("--level", default="char", choices=["char", "word", "byte"],
                   help="token level when using SimpleTokenizer; 'byte' forces byte-level")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=500,
                   help="max training steps (overrides epochs if set)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear LR warmup steps (0 to disable)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/ch10_gpt.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device

    # Build ids
    text = load_texts(args.data)
    if args.level == "byte":
        ids_info = build_ids_byte_level(text)
    else:
        ids_info = build_ids_with_tokenizer(text, level=args.level)

    ds = LMSequenceDataset(ids_info.ids, block_size=args.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model config & model
    cfg = GPTConfig(
        vocab_size=ids_info.vocab_size,
        block_size=args.block_size,
        d_model=256,
        n_head=4,
        n_layer=4,
        d_ff=1024,
        dropout=0.1,
        pos_type="learned",
        tie_weights=True,
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = None
    if args.warmup_steps > 0:
        # Linear warmup from 0 -> 1 over warmup_steps
        def lr_lambda(step: int) -> float:
            return min(1.0, (step + 1) / float(args.warmup_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print("Device:", device)
    print("Config:", asdict(cfg))
    print("Dataset tokens:", ds.ids.numel())

    step = 0
    t0 = time()
    model.train()
    for epoch in range(max(1, args.epochs)):
        for x, y in dl:
            if args.steps and step >= args.steps:
                break
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, loss = model(x, targets=y, pad_id=ids_info.pad_id)
            assert loss is not None
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if step % 50 == 0:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"step {step:5d} lr {lr_now:.5f} "
                    f"loss {loss.detach().item():.4f}"
                )
            step += 1
        if args.steps and step >= args.steps:
            break

    dt = time() - t0
    print(f"Done. steps={step} time={dt:.1f}s")

    # Save checkpoint
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "config": asdict(cfg),
        "model_state": model.state_dict(),
    }
    # Save tokenizer metadata if available for easier sampling later
    if ids_info.id_to_token is not None:
        ckpt["tokenizer"] = {
            "level": ids_info.level,
            "id_to_token": ids_info.id_to_token,
            "pad_id": ids_info.pad_id,
            "unk_id": ids_info.unk_id,
        }
    torch.save(ckpt, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
