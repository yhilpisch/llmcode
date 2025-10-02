"""Minimal environment and device sanity check.

Run with: python -m code.env_check
"""
from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    print("== Environment ==")
    print("Python:", platform.python_version())
    print("Platform:", platform.platform())
    print("Executable:", sys.executable)
    print("CWD:", os.getcwd())

    try:
        import torch  # type: ignore

        print("\n== PyTorch ==")
        print("torch:", torch.__version__)
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        print("CUDA available:", cuda)
        if cuda:
            print("CUDA device count:", torch.cuda.device_count())
            if torch.cuda.device_count() > 0:
                print("CUDA device 0:", torch.cuda.get_device_name(0))
        print("MPS available:", bool(mps and torch.backends.mps.is_available()))
    except Exception as e:  # pragma: no cover - diagnostics only
        print("\nPyTorch not installed or not importable:", e)


if __name__ == "__main__":
    main()

