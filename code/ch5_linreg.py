"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Minimal linear regression training in PyTorch (Chapter 5).

Run:
  python code/ch5_linreg.py --device auto --epochs 400
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    epochs: int = 400
    lr: float = 3e-2
    n: int = 128
    seed: int = 42
    device: str = "auto"  # cpu|cuda|mps|auto


def make_data(
    cfg: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Fix RNG for reproducibility across devices
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    w_true = torch.tensor([2.0, -3.5])
    b_true = torch.tensor(0.5)
    # Draw features and small Gaussian noise on target
    X = torch.randn(cfg.n, 2, generator=g).to(device)
    noise = 0.1 * torch.randn(cfg.n, generator=g).to(device)
    y = (X @ w_true.to(device)) + b_true.to(device) + noise
    return X, y, w_true.to(device), b_true.to(device)


def train(cfg: Config) -> None:
    # Pick device lazily to match user selection
    device = pick_device() if cfg.device == "auto" else torch.device(cfg.device)
    X, y, w_true, b_true = make_data(cfg, device)

    model = torch.nn.Linear(2, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    for step in range(cfg.epochs + 1):
        # Usual gradient-descent step: zero, forward, loss, backward, update
        opt.zero_grad()
        pred = model(X).squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")

    w_learned = model.weight.detach().squeeze(0)
    b_learned = model.bias.detach().squeeze(0)
    print("true  w:", w_true.cpu().tolist(), " b:", float(b_true))
    print("learn w:", w_learned.cpu().tolist(), " b:", float(b_learned))


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--device", default="auto")
    ns = ap.parse_args()
    return Config(epochs=ns.epochs, lr=ns.lr, device=ns.device)


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
