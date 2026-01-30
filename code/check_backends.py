"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Print available PyTorch backends and basic device info.

Run with: python -m code.check_backends
"""

from __future__ import annotations

def main() -> None:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        print("PyTorch not installed:", e)
        return

    has_mps_backend = getattr(torch.backends, "mps", None)
    # Guard against missing PyTorch by echoing version early
    print("torch:", torch.__version__)
    has_cuda = torch.cuda.is_available()
    has_mps = bool(has_mps_backend and torch.backends.mps.is_available())
    # Report CUDA capability first for people with multiple GPUs
    print("CUDA available:", has_cuda)
    if has_cuda:
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            print(f"  [{i}]", name)
    # Show Apple MPS status as a secondary hardware target
    print("MPS available:", has_mps)
    device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
    # Preferred device ordering mirrors the training scripts
    print("Preferred device:", device)


if __name__ == "__main__":
    main()
