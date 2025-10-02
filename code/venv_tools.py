"""Virtual environment helpers used in Chapter 2.

Usage:
  python -m code.venv_tools info
  python -m code.venv_tools create .venv
"""
from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


def info() -> None:
    print("== Python & Environment ==")
    print("Executable:", sys.executable)
    print("Prefix:", sys.prefix)
    venv = os.environ.get("VIRTUAL_ENV") or (".venv" if ".venv" in sys.executable else "")
    print("VIRTUAL_ENV:", venv or "(not active)")
    site_dirs = site.getsitepackages()
    sp = ", ".join(p for p in site_dirs if ".venv" in p) or ", ".join(site_dirs)
    print("site-packages:", sp)


def create(path: str = ".venv") -> None:
    """Create a virtual environment at `path` if it doesn't exist.

    This is a convenience wrapper around: `python -m venv <path>`.
    """
    p = Path(path)
    if p.exists():
        print(f"Environment already exists at {p}")
        return
    print("Creating venv:", p)
    subprocess.check_call([sys.executable, "-m", "venv", str(p)])
    print("Created. To activate:")
    if os.name == "nt":
        print(rf"  .\{p}\Scripts\Activate.ps1  # PowerShell")
    else:
        print(f"  source {p}/bin/activate")


def remove(path: str = ".venv") -> None:
    p = Path(path)
    if not p.exists():
        print("No such environment:", p)
        return
    print("Removing venv:", p)
    shutil.rmtree(p)
    print("Removed.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print("Usage: python -m code.venv_tools [info|create|remove] [path]")
        return
    cmd = argv.pop(0)
    if cmd == "info":
        info()
    elif cmd == "create":
        create(argv[0] if argv else ".venv")
    elif cmd == "remove":
        remove(argv[0] if argv else ".venv")
    else:
        print("Unknown command:", cmd)
        sys.exit(2)


if __name__ == "__main__":
    main()
