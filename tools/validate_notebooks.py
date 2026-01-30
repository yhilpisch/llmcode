#!/usr/bin/env python3
import argparse
import ast
import importlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple


def fmt_duration(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds * 1e6:.1f}us"


def normalize_cell_ids(nb: "nbformat.NotebookNode") -> int:  # type: ignore[name-defined]
    changed = 0
    seen: Set[str] = set()
    for i, cell in enumerate(nb.cells):
        cid = cell.get("id")
        if not cid or cid in seen:
            new_id = f"cell-{i:04d}"
            j = 0
            candidate = new_id
            while candidate in seen:
                j += 1
                candidate = f"{new_id}-{j}"
            cell["id"] = candidate
            changed += 1
            cid = candidate
        seen.add(str(cid))
    return changed


def check_line_lengths(nb: "nbformat.NotebookNode", max_len: int) -> List[Tuple[int, int]]:  # type: ignore[name-defined]
    hits: List[Tuple[int, int]] = []
    for idx, cell in enumerate(nb.cells):
        if getattr(cell, "cell_type", None) != "code":
            continue
        source = cell.get("source", "") or ""
        for ln, line in enumerate(str(source).splitlines(), start=1):
            if len(line) > max_len:
                hits.append((idx + 1, ln))
    return hits


def extract_imports_from_code(code: str) -> Set[str]:
    modules: Set[str] = set()
    try:
        tree = ast.parse(code)
    except Exception:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def scan_imports(nb: "nbformat.NotebookNode") -> Set[str]:  # type: ignore[name-defined]
    imports: Set[str] = set()
    for cell in nb.cells:
        if getattr(cell, "cell_type", None) == "code":
            code = cell.get("source", "") or ""
            imports.update(extract_imports_from_code(code))
    return imports


OPTIONAL_IMPORTS = {"matplotlib", "fastapi", "pydantic", "streamlit", "sentencepiece", "tokenizers"}


REPO_ROOT = Path(__file__).resolve().parents[1]


def probe_imports(mods: Set[str]) -> Tuple[bool, List[str], List[str]]:
    failures = []
    optional_missing = []
    for name in sorted(mods):
        # Skip probing local modules that live under code/ (they will import during execution)
        if (REPO_ROOT / "code" / f"{name}.py").exists() or (REPO_ROOT / "code" / name / "__init__.py").exists():
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            line = f"{name}: {type(exc).__name__}: {exc}"
            if name in OPTIONAL_IMPORTS:
                optional_missing.append(line)
            else:
                failures.append(line)
    return (len(failures) == 0, failures, optional_missing)


def execute_notebook(nb: "nbformat.NotebookNode", path: Path, timeout: float, kernel_name: Optional[str] = None) -> Tuple[bool, str]:  # type: ignore[name-defined]
    try:
        import nbformat  # type: ignore
        from nbclient import NotebookClient  # type: ignore
        from nbclient.exceptions import CellExecutionError  # type: ignore
    except Exception:
        return False, "nbclient not installed (pip install nbclient nbformat)"

    # Resolve kernel name: CLI > notebook metadata > default 'python3'
    meta_kernel = None
    try:
        meta_kernel = nb.metadata.get("kernelspec", {}).get("name")  # type: ignore[assignment]
    except Exception:
        meta_kernel = None
    resolved_kernel = kernel_name or meta_kernel or "python3"

    client = NotebookClient(
        nb,
        timeout=int(timeout),
        kernel_name=str(resolved_kernel),
        allow_errors=False,
        record_timing=False,
    )
    try:
        client.execute()
        return True, "OK"
    except CellExecutionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_notebook(path: Path) -> Tuple[bool, Optional["nbformat.NotebookNode"], Optional[str]]:  # type: ignore[name-defined]
    try:
        import nbformat  # type: ignore
    except Exception as exc:
        return False, None, f"nbformat not available: {exc}"
    try:
        nb = nbformat.read(str(path), as_version=4)
        return True, nb, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def normalize_notebook(path: Path, nb: "nbformat.NotebookNode") -> Tuple[bool, Optional["nbformat.NotebookNode"], Optional[str], bool]:  # type: ignore[name-defined]
    try:
        from nbformat.validator import normalize as nb_normalize  # type: ignore
    except Exception as exc:
        return False, nb, f"normalize unavailable: {exc}", False

    try:
        import nbformat  # type: ignore
    except Exception as exc:
        return False, nb, f"nbformat not available: {exc}", False

    before = nbformat.writes(nb)
    try:
        changes, normalized_dict = nb_normalize(nb)
    except Exception as exc:
        return False, nb, f"normalize failed: {exc}", False

    normalized_nb = nbformat.from_dict(normalized_dict)
    after = nbformat.writes(normalized_nb)
    changed = (before != after) or bool(changes)
    return True, normalized_nb, None, changed


def print_stage(label: str, status: str, detail: str) -> None:
    formatted_label = f"{label}:"
    print(f"  • {formatted_label:<12} {status:<7} ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and execute Jupyter notebooks under a directory.")
    parser.add_argument("--path", default="notebooks", help="Root directory to search for .ipynb files (default: notebooks)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-notebook execution timeout in seconds (default: 300)")
    parser.add_argument("--pattern", default="*.ipynb", help="Glob pattern for notebooks (default: *.ipynb)")
    parser.add_argument("--kernel", default=None, help="Kernel name to use (default: auto-detect)")
    parser.add_argument("--max-len", type=int, default=85, help="Max line length for code cells (default: 85)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--normalize", action="store_true", help="Normalize notebooks before validation")
    parser.add_argument("--normalize-ids", action="store_true", help="Normalize missing/duplicate cell ids")
    parser.add_argument("--write-normalized", action="store_true", help="Write normalized notebooks back in place")
    parser.add_argument("--execute", action="store_true", help="Execute notebooks after validation")
    parser.add_argument("--skip-execute", action="store_true", help="Skip executing notebooks")
    args = parser.parse_args()
    do_execute = args.execute and not args.skip_execute

    root = Path(args.path)
    if not root.exists():
        print(f"No such directory: {root}")
        return 1

    notebooks = sorted([p for p in root.rglob(args.pattern) if p.is_file()])
    if not notebooks:
        print("No notebooks found.")
        return 0

    # Ensure project root on PYTHONPATH for local imports; avoid adding code/ to prevent shadowing stdlib 'code'
    project_root = Path(__file__).resolve().parents[1]
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(project_root),
        os.environ.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    total_start = time.perf_counter()
    failures = 0
    count = len(notebooks)

    for idx, nb_path in enumerate(notebooks, start=1):
        print(f"[{idx}/{count}] Validating {nb_path}")

        nb_obj = None
        load_msg: Optional[str] = None

        # Load notebook (needed for further steps)
        load_start = time.perf_counter()
        ok, nb_obj, load_msg = load_notebook(nb_path)
        load_duration = time.perf_counter() - load_start
        if not ok or nb_obj is None:
            print_stage("Structure", "FAIL", fmt_duration(load_duration))
            if load_msg:
                print(f"    ↳ {load_msg}")
            failures += 1
            if args.fail_fast:
                break
            print()
            continue
        total_item_time = load_duration

        # Normalize stage (nbformat)
        norm_changed = False
        if args.normalize:
            n0 = time.perf_counter()
            norm_ok, nb_obj, norm_msg, changed = normalize_notebook(nb_path, nb_obj)
            n1 = time.perf_counter()
            total_item_time += n1 - n0
            if norm_ok and nb_obj is not None:
                norm_changed = changed
                print_stage("Normalize", "OK", f"{fmt_duration(n1 - n0)}")
                if changed:
                    print("    ↳ notebook normalized")
            else:
                print_stage("Normalize", "FAIL", fmt_duration(n1 - n0))
                if norm_msg:
                    print(f"    ↳ {norm_msg}")
                failures += 1
                if args.fail_fast:
                    break
                print()
                continue
        else:
            print_stage("Normalize", "SKIP", "disabled")

        # Normalize cell ids
        id_changes = 0
        if args.normalize_ids and nb_obj is not None:
            i0 = time.perf_counter()
            id_changes = normalize_cell_ids(nb_obj)
            i1 = time.perf_counter()
            total_item_time += i1 - i0
            if id_changes:
                print_stage("NormIDs", "OK", f"{fmt_duration(i1 - i0)}")
                print(f"    ↳ normalized {id_changes} cell id(s)")
            else:
                print_stage("NormIDs", "OK", f"{fmt_duration(i1 - i0)}")
        else:
            print_stage("NormIDs", "SKIP", "disabled")

        # Write normalized notebook if requested
        if args.write_normalized and nb_obj is not None and (norm_changed or id_changes):
            try:
                import nbformat  # type: ignore
                nbformat.write(nb_obj, str(nb_path))
                print("    ↳ wrote normalized notebook")
            except Exception as exc:
                failures += 1
                print(f"    ↳ write failed: {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    break

        # Structure validation (after potential normalization)
        s0 = time.perf_counter()
        structure_ok = True
        structure_msg: Optional[str] = None
        try:
            import nbformat  # type: ignore
            nbformat.validate(nb_obj)  # type: ignore[arg-type]
        except Exception as exc:
            structure_ok = False
            structure_msg = f"{type(exc).__name__}: {exc}"
        s1 = time.perf_counter()
        total_item_time += s1 - s0
        print_stage("Structure", "OK" if structure_ok else "FAIL", fmt_duration(s1 - s0))
        if not structure_ok:
            if structure_msg:
                print(f"    ↳ {structure_msg}")
            failures += 1
            if args.fail_fast:
                break
            print()
            continue

        # Line-length check (code cells only)
        l0 = time.perf_counter()
        line_hits = check_line_lengths(nb_obj, args.max_len)
        l1 = time.perf_counter()
        total_item_time += l1 - l0
        line_ok = len(line_hits) == 0
        if line_ok:
            print_stage("LineLen", "OK", fmt_duration(l1 - l0))
        else:
            preview = ", ".join([f"c{c}l{l}" for c, l in line_hits[:6]])
            if len(line_hits) > 6:
                preview += ", ..."
            print_stage("LineLen", "FAIL", fmt_duration(l1 - l0))
            print(f"    ↳ {len(line_hits)} long line(s) > {args.max_len} (e.g., {preview})")
            failures += 1
            if args.fail_fast:
                break

        # Imports: scan and probe
        i0 = time.perf_counter()
        imports = scan_imports(nb_obj)
        i1 = time.perf_counter()
        p0 = time.perf_counter()
        probe_ok, probe_failures, optional_missing = probe_imports(imports)
        p1 = time.perf_counter()
        imports_ok = probe_ok  # scan returns empty set if nbformat missing
        total_item_time += (i1 - i0) + (p1 - p0)
        print_stage("Imports", "OK" if imports_ok else "FAIL", f"{fmt_duration(i1 - i0)} scan, {fmt_duration(p1 - p0)} probe")
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
        if do_execute:
            e0 = time.perf_counter()
            ok_exec, msg = execute_notebook(nb_obj, nb_path, args.timeout, kernel_name=args.kernel)
            e1 = time.perf_counter()
            total_item_time += e1 - e0
            if not ok_exec and "nbclient not installed" in msg:
                print_stage("Execute", "SKIP", fmt_duration(e1 - e0))
                print(f"    ↳ {msg}")
            else:
                print_stage("Execute", "OK" if ok_exec else "FAIL", fmt_duration(e1 - e0))
                if not ok_exec:
                    print(f"    ↳ {msg}")
                    failures += 1
                    if args.fail_fast:
                        break
        else:
            print_stage("Execute", "SKIP", "disabled")

        print_stage("Total", "—", fmt_duration(total_item_time))
        print()

    total_time = time.perf_counter() - total_start
    print(f"Executed {count} notebooks in {total_time:.2f}s. Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
