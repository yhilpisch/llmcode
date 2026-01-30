"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Tiny matmul timer to sanity-check device speed.

Example:
  python -m code.bench_timer --device auto --size 2048 --repeats 5
"""

from __future__ import annotations

import argparse
import time


def pick_device(torch):  # type: ignore
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        print("PyTorch not installed:", e)
        return

    # Read basic matmul settings from CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="cpu|cuda|mps|auto")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    # Resolve device and create square matrices
    device = pick_device(torch) if args.device == "auto" else args.device
    N = args.size
    x = torch.randn(N, N, device=device)
    y = torch.randn(N, N, device=device)

    # Warmup for CUDA/MPS
    for _ in range(2):
        _ = x @ y
        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else None

    times = []
    for _ in range(args.repeats):
        # Time a single matmul and sync to measure wall time
        t0 = time.time()
        z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            # best-effort; MPS ops often synchronize implicitly on tensor access
            _ = z.cpu()
        times.append(time.time() - t0)

    print(
        {
            "device": device,
            "size": N,
            "repeats": args.repeats,
            "ms_mean": round(1000 * sum(times) / len(times), 2),
            "ms_min": round(1000 * min(times), 2),
        }
    )


if __name__ == "__main__":
    main()
