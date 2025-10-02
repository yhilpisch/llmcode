"""Self-contained example workspace and sample texts.

Use this when readers don't have the book repo. It creates a temporary
folder in the current working directory, fills it with small sample
text files, and (optionally) cleans them up when done.

Usage (CLI):
  python -m code.example_data create --defaults [--keep]
  python -m code.example_data path     # print last created path
  python -m code.example_data cleanup <path>

Usage (Python):
  from code.example_data import ExampleWorkspace
  with ExampleWorkspace().create_defaults() as ws:
      print(ws.root)  # use ws.root / files inside
      ...
"""
from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

_LAST_PATH_FILE = Path(".example_workspace_path")


DEFAULT_TEXTS = {
    "philosophy.txt": (
        "We are what we repeatedly do. Excellence, then, is not an act "
        "but a habit. Questions sharpen knowledge; curiosity sustains it.\n"
    ),
    "science.txt": (
        "Science is a way of thinking much more than it is a body of facts. "
        "Small experiments illuminate large ideas.\n"
    ),
    "poetry.txt": (
        "The model dreams in tokens and time,\n"
        "A lantern of vectors that learn to rhyme.\n"
    ),
}


@dataclass
class ExampleWorkspace:
    base_dir: Path = Path.cwd()
    name: str | None = None
    cleanup_on_exit: bool = True

    def __post_init__(self) -> None:
        if self.name is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.name = f"examples-{stamp}"
        self.root = self.base_dir / self.name  # type: ignore[attr-defined]

    def create(self) -> "ExampleWorkspace":
        self.root.mkdir(parents=True, exist_ok=True)
        _LAST_PATH_FILE.write_text(str(self.root))
        return self

    def create_defaults(self) -> "ExampleWorkspace":
        self.create()
        for fname, text in DEFAULT_TEXTS.items():
            (self.root / fname).write_text(text)
        return self

    def add_text(self, filename: str, content: str) -> Path:
        p = self.root / filename
        p.write_text(content)
        return p

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        if _LAST_PATH_FILE.exists():
            _LAST_PATH_FILE.unlink()

    # Context manager API
    def __enter__(self) -> "ExampleWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self.cleanup_on_exit:
            self.cleanup()


def _cmd_create(args: argparse.Namespace) -> None:
    ws = ExampleWorkspace(cleanup_on_exit=not args.keep)
    (ws.create_defaults())
    print(ws.root)
    if not args.keep:
        print("(Temporary; will be cleaned up when used via context manager or explicitly)")


def _cmd_path(_: argparse.Namespace) -> None:
    if _LAST_PATH_FILE.exists():
        print(_LAST_PATH_FILE.read_text())
    else:
        print("No workspace recorded. Use 'create' first.")


def _cmd_cleanup(args: argparse.Namespace) -> None:
    target = Path(args.path).resolve()
    if not target.exists():
        print("No such path:", target)
        return
    shutil.rmtree(target)
    if _LAST_PATH_FILE.exists():
        try:
            last = Path(_LAST_PATH_FILE.read_text().strip())
            if last == target:
                _LAST_PATH_FILE.unlink()
        except Exception:
            _LAST_PATH_FILE.unlink(missing_ok=True)  # type: ignore[attr-defined]
    print("Removed:", target)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="code.example_data", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="create a workspace and write default texts")
    c.add_argument("--keep", action="store_true", help="do not auto-clean later")
    c.set_defaults(func=_cmd_create)

    sub.add_parser("path", help="print last workspace path").set_defaults(func=_cmd_path)

    d = sub.add_parser("cleanup", help="remove a workspace path")
    d.add_argument("path", help="path to workspace directory")
    d.set_defaults(func=_cmd_cleanup)

    ns = p.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()

