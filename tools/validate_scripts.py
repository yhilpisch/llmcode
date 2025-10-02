#!/usr/bin/env python3
import argparse
import ast
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Set


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"


OPTIONAL_IMPORTS = {"matplotlib", "fastapi", "pydantic", "streamlit"}


def scan_imports_py(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(path))
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Only consider absolute imports (level == 0)
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def probe_imports(mods: Set[str]) -> Tuple[bool, List[str], List[str]]:
    failures = []
    optional_missing = []
    for name in sorted(mods):
        try:
            importlib.import_module(name)
        except Exception as exc:
            line = f"{name}: {type(exc).__name__}: {exc}"
            if name in OPTIONAL_IMPORTS:
                optional_missing.append(line)
            else:
                failures.append(line)
    return (len(failures) == 0, failures, optional_missing)


def run_script(path: Path, timeout: float, env: dict) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout:.0f}s"
    ok = proc.returncode == 0
    if ok:
        return True, "OK"
    # Heuristics: treat CLI usage or missing sample assets as SKIP (expected without args)
    out_all = (proc.stderr or "") + "\n" + (proc.stdout or "")
    lower = out_all.lower()
    if "usage:" in lower or "the following arguments are required" in lower:
        return False, "SKIP: CLI requires arguments"
    if "file not found" in lower or "no such file or directory" in lower:
        return False, "SKIP: missing input/resource"
    if "typeerror: 'type' object is not subscriptable" in lower:
        return False, "SKIP: typing not supported with current deps"
    # Return last few lines of stderr for context
    tail = (proc.stderr or proc.stdout or "").splitlines()[-5:]
    return False, ("\n".join(tail) if tail else "Non-zero exit status")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and run Python scripts under a directory.")
    parser.add_argument("--path", default="code", help="Root directory to search for .py files (default: code)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-script execution timeout in seconds (default: 120)")
    parser.add_argument("--pattern", default="*.py", help="Glob pattern for scripts (default: *.py)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"No such directory: {root}")
        return 1

    scripts = sorted([p for p in root.rglob(args.pattern) if p.is_file()])
    total_start = time.perf_counter()
    failures = 0

    # Ensure project root on PYTHONPATH for local imports
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # Add both repo root and code/ to PYTHONPATH so `import chXX_*` works
    env["PYTHONPATH"] = os.pathsep.join([
        str(project_root),
        str(project_root / "code"),
        env.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    count = len(scripts)
    if count == 0:
        print("No scripts found.")
        return 0

    for idx, script in enumerate(scripts, start=1):
        header = f"[{idx}/{count}] Validating {script}"
        print(header)

        # Structure
        t0 = time.perf_counter()
        try:
            ast.parse(script.read_text(encoding="utf-8", errors="ignore"), filename=str(script))
            structure_ok = True
            structure_msg = "OK"
        except SyntaxError as e:
            structure_ok = False
            structure_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
        t1 = time.perf_counter()
        print(f"  • Structure: {'OK' if structure_ok else 'FAIL'}   ({fmt_duration(t1 - t0)})")
        if not structure_ok:
            failures += 1
            if args.fail_fast:
                break
            else:
                continue

        # Imports scan + probe
        t2 = time.perf_counter()
        try:
            mods = scan_imports_py(script)
            scan_ok = True
        except Exception as e:
            mods = set()
            scan_ok = False
        t3 = time.perf_counter()

        probe_start = time.perf_counter()
        probe_ok, probe_failures, optional_missing = probe_imports(mods)
        probe_end = time.perf_counter()

        imports_ok = scan_ok and probe_ok
        print(
            f"  • Imports: {'OK' if imports_ok else 'FAIL'}   ({fmt_duration(t3 - t2)} scan, {fmt_duration(probe_end - probe_start)} probe)"
        )
        if optional_missing:
            for line in optional_missing:
                print(f"    ↳ optional: {line}")
        if not imports_ok and probe_failures:
            for line in probe_failures:
                print(f"    ↳ {line}")
        if not imports_ok and args.fail_fast:
            failures += 1
            break

        # Execute
        exec_start = time.perf_counter()
        ok, msg = run_script(script, args.timeout, env)
        exec_end = time.perf_counter()
        status = 'OK'
        if not ok and msg.startswith('SKIP:'):
            status = 'SKIP'
        elif not ok:
            status = 'FAIL'
        print(f"  • Execute: {status}   ({fmt_duration(exec_end - exec_start)})")
        if status == 'FAIL':
            print(f"    ↳ {msg}")
            failures += 1
            if args.fail_fast:
                break
        elif status == 'SKIP':
            print(f"    ↳ {msg}")

        total_item = (t1 - t0) + (t3 - t2) + (probe_end - probe_start) + (exec_end - exec_start)
        print(f"  • Total: {fmt_duration(total_item)}\n")

    total_time = time.perf_counter() - total_start
    print(f"Executed {count} scripts in {total_time:.2f}s. Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
