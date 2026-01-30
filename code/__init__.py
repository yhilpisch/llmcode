"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Project code package (mirrors stdlib ``code`` attributes for compatibility).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from types import ModuleType
from typing import Any

_STDLIB_MODULE: ModuleType | None = None

def _load_stdlib_code() -> ModuleType | None:
    """Load the standard library ``code`` module.

    This avoids issues because this package shadows the stdlib name.
    """
    try:
        stdlib_dir = sysconfig.get_paths().get("stdlib")
        if not stdlib_dir:
            return None
        stdlib_code_path = os.path.join(stdlib_dir, "code.py")
        if not os.path.exists(stdlib_code_path):
            return None
        spec = importlib.util.spec_from_file_location(
            "_stdlib_code",
            stdlib_code_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[assignment]
        sys.modules.setdefault("_stdlib_code", module)
        return module
    except Exception:
        return None


_STDLIB_MODULE = _load_stdlib_code()

if _STDLIB_MODULE is not None:
    stdlib_all = getattr(_STDLIB_MODULE, "__all__", None)
    names = stdlib_all if isinstance(stdlib_all, (list, tuple)) else [
        name for name in dir(_STDLIB_MODULE) if not name.startswith("_")
    ]
    globals().update({name: getattr(_STDLIB_MODULE, name) for name in names})
    __all__ = list(names)  # type: ignore[assignment]
else:
    __all__: list[str] = []


def __getattr__(name: str) -> Any:
    if _STDLIB_MODULE is not None and hasattr(_STDLIB_MODULE, name):
        return getattr(_STDLIB_MODULE, name)
    raise AttributeError(f"module 'code' has no attribute {name!r}")
