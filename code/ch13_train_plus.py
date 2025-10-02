from __future__ import annotations

"""Improved training loop with AMP, clipping, scheduling, and validation.

Chapter 13 builds on Chapter 10's trainer:
- Optional AMP (mixed precision) for speed on GPU.
- Gradient clipping to tame spikes.
- Warmup + cosine LR schedule.
- Gradient accumulation to simulate larger batches.
- Train/val split and best-checkpoint saving.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import time
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Make the "code/" directory importable when running as a script so this file
# works when executed as `python code/ch13_train_plus.py` from the repo root.
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch10_data import (  # type: ignore
    LMSequenceDataset,
    build_ids_with_tokenizer,
    build_ids_byte_level,
    load_texts,
)
from ch13_schedules import warmup_cosine_lr  # type: ignore
from ch11_metrics import perplexity  # type: ignore


def auto_device() -> str:
    """Return the best available device string ('cuda'|'mps'|'cpu').

    - CUDA if available (best throughput for AMP)
    - MPS (Apple Silicon GPU) if available
    - CPU otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loaders(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    val_ratio: float,
    split: str = "contiguous",
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from a 1-D id stream.

    We slice the id stream into (x,y) windows and split into train/val
    partitions with a small held-out set for validation.
    """
    if split == "contiguous":
        N = ids.numel()
        cut = max(block_size + 1, int(N * (1 - val_ratio)))
        train_ids = ids[:cut]
        val_ids = ids[max(0, cut - block_size - 1):]
        train_ds = LMSequenceDataset(train_ids, block_size)
        val_ds = LMSequenceDataset(val_ids, block_size)
    else:
        ds = LMSequenceDataset(ids, block_size)
        n_val = max(1, int(len(ds) * val_ratio))
        n_train = max(0, len(ds) - n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl


def main() -> None:
    p = argparse.ArgumentParser(description="Improved GPT trainer (Chapter 13)")
    p.add_argument("--data", nargs="*", help="text file(s)")
    p.add_argument("--level", default="char", choices=["char", "word", "byte"])
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1, help="grad accumulation")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--split", choices=["contiguous","random"], default="contiguous",
                   help="how to split the id stream before windowing")
    p.add_argument(
        "--val-interval-steps",
        type=int,
        default=0,
        help="run validation every N optimizer steps (0 = only at epoch end)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--cosine-min-ratio", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/ch13_gpt_best.pt")
    p.add_argument("--resume", type=str, default="", help="resume from checkpoint .pt")
    args = p.parse_args()

    print("[1/8] Seeding and device selection")
    torch.manual_seed(args.seed)
    device = auto_device() if args.device == "auto" else args.device

    # Data → ids
    print("[2/8] Loading text files")
    text = load_texts(args.data)
    print(f"[3/8] Building token ids (level={args.level})")
    if args.level == "byte":
        ids_info = build_ids_byte_level(text)
    else:
        ids_info = build_ids_with_tokenizer(text, level=args.level)
    print("[4/8] Creating train/val loaders")
    train_dl, val_dl = make_loaders(
        ids_info.ids,
        args.block_size,
        args.batch_size,
        args.val_ratio,
        args.split,
    )

    # Model (optionally resume from checkpoint config/weights)
    print("[5/8] Building model" + (" (resume)" if args.resume else ""))
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        cfg = GPTConfig(**ckpt["config"])  # enforce saved config
        if cfg.block_size != args.block_size:
            print(
                "NOTE: overriding block_size",
                args.block_size,
                "->",
                cfg.block_size,
                "from checkpoint",
            )
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])  # type: ignore
        print("Resumed weights from:", args.resume)
    else:
        cfg = GPTConfig(
            vocab_size=ids_info.vocab_size,
            block_size=args.block_size,
            d_model=256,
            n_head=4,
            n_layer=2,
            d_ff=int(1536 / 2),
            dropout=0.1,
        )
        model = GPT(cfg).to(device)
    print("[6/8] Optimizer, scheduler, AMP")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine_lr(opt, args.warmup_steps, args.steps, args.cosine_min_ratio)
    amp_enabled = (args.amp and device == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)  # torch>=2.4
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("Device:", device)
    print("Config:", asdict(cfg))
    print("Train steps:", args.steps, "accum:", args.accum_steps)

    print("[7/8] Training with progress bars")
    # Small helper to run validation and return (H, PPL). We compute the CE
    # loss on CPU float32 for numeric stability across backends (CUDA/MPS/CPU).
    def _run_validation() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc="val", unit="batch", leave=False):
                xb = xb.to(device); yb = yb.to(device)
                # Forward on device, then compute CE on CPU for numeric stability
                logits, _ = model(xb, targets=None)
                B, T, V = logits.shape
                lf = logits.detach().to("cpu", dtype=torch.float32).reshape(B*T, V)
                yf = yb.detach().to("cpu", dtype=torch.long).reshape(B*T)
                batch_loss = torch.nn.functional.cross_entropy(lf, yf, reduction="sum")
                total_loss += float(batch_loss.item())
                total_tokens += int(B*T)
        import math
        H = total_loss / max(1, total_tokens)
        PPL = math.exp(H)
        return H, PPL
    model.train()
    best_val = float("inf")
    step = 0
    t0 = time()
    accum = 0
    pbar = tqdm(total=args.steps, desc="train", unit="step")
    last_val_H: float | None = None
    while step < args.steps:
        for xb, yb in train_dl:
            if step >= args.steps:
                break
            xb = xb.to(device); yb = yb.to(device)
            # Autocast only on CUDA; MPS/CPU run in full precision
            ac = (
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if amp_enabled
                else nullcontext()
            )
            with ac:
                _, loss = model(
                    xb,
                    targets=yb,
                    pad_id=ids_info.pad_id,
                )
            assert loss is not None
            # Backward with AMP scale on CUDA (no-op elsewhere)
            _scaled = scaler.scale(loss) if scaler.is_enabled() else loss
            _scaled.backward()
            accum += 1
            if accum % args.accum_steps == 0:
                if args.clip_grad and args.clip_grad > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.clip_grad,
                    )
                if scaler.is_enabled():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                lr_now = opt.param_groups[0]["lr"]
                train_loss = float(loss.detach().item())
                # Optional mid-epoch validation
                if args.val_interval_steps and (step % args.val_interval_steps == 0):
                    H, PPL = _run_validation()
                    last_val_H = H
                    pbar.write(
                        f"val@{step}: H={H:.4f} PPL={PPL:.2f}"
                    )
                    if H < best_val:
                        best_val = H
                        out = Path(args.save)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        ckpt = {
                            "config": asdict(cfg),
                            "model_state": model.state_dict(),
                        }
                        # Save tokenizer metadata if present
                        if ids_info.id_to_token is not None:
                            ckpt["tokenizer"] = {
                                "level": ids_info.level,
                                "id_to_token": ids_info.id_to_token,
                                "pad_id": ids_info.pad_id,
                                "unk_id": ids_info.unk_id,
                            }
                        torch.save(ckpt, out)
                        pbar.write(f"Saved best: {out}")
                    model.train()
                val_str = (
                    f"{last_val_H:.4f}" if last_val_H is not None else "-"
                )
                pbar.set_postfix(
                    lr=f"{lr_now:.5f}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=val_str,
                )
                pbar.update(1)

        # End epoch validation unless we already do it mid-epoch
        if not args.val_interval_steps:
            H, PPL = _run_validation()
            last_val_H = H
            pbar.write(f"val: H={H:.4f} PPL={PPL:.2f}")
            if H < best_val:
                best_val = H
                out = Path(args.save)
                out.parent.mkdir(parents=True, exist_ok=True)
                ckpt = {"config": asdict(cfg), "model_state": model.state_dict()}
                if ids_info.id_to_token is not None:
                    ckpt["tokenizer"] = {
                        "level": ids_info.level,
                        "id_to_token": ids_info.id_to_token,
                        "pad_id": ids_info.pad_id,
                        "unk_id": ids_info.unk_id,
                    }
                torch.save(ckpt, out)
                pbar.write(f"Saved best: {out}")
            model.train()
    pbar.close()

    print(f"[8/8] Done in {time()-t0:.1f}s. Best val H={best_val:.4f}")


if __name__ == "__main__":
    main()
