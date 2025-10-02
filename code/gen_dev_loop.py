"""Generate a simple dev loop diagram (edit → run → iterate → commit).

If `graphviz` is installed, we render SVG directly; else we write DOT.
"""
from __future__ import annotations

from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

dot_content = r"""
digraph DevLoop {
  rankdir=LR;
  node [shape=box, style=rounded, color="#4B5563", fontname="Helvetica"];
  edge [color="#6B7280"];

  edit    [label="Edit\n(code / text)"];
  run     [label="Run\n(scripts / tests)"];
  iterate [label="Iterate\n(tune / refactor)"];
  commit  [label="Commit\n(Git / PR)"];

  edit -> run -> iterate -> edit;
  iterate -> commit;
}
"""

def main() -> None:
    try:
        from graphviz import Source  # type: ignore

        s = Source(dot_content)
        out = s.render(filename=str(FIG_DIR / "dev-loop"), format="svg", cleanup=True)
        print("Wrote:", out)
    except Exception as e:
        dot_path = FIG_DIR / "dev-loop.dot"
        dot_path.write_text(dot_content)
        print("graphviz not available (", e, ")\nWrote DOT:", dot_path)
        print("Render manually with:\n  dot -Tsvg figures/dev-loop.dot -o figures/dev-loop.svg")


if __name__ == "__main__":
    main()

