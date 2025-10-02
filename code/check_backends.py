"""Print available PyTorch backends and basic device info.

Run with: python -m code.check_backends
"""
from __future__ import annotations

def main() -> None:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        print("PyTorch not installed:", e)
        return

    print("torch:", torch.__version__)
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print("CUDA available:", has_cuda)
    if has_cuda:
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}]", torch.cuda.get_device_name(i))
    print("MPS available:", has_mps)
    device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
    print("Preferred device:", device)


if __name__ == "__main__":
    main()

