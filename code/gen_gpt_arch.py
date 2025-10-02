"""Generate a GPT architecture diagram for Chapter 9.

If `graphviz` Python package is installed, renders SVG directly to
`figures/ch09-gpt-arch.svg`. Else, writes a DOT file and prints instructions
to render it with `dot`.
"""
from __future__ import annotations

from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

dot = r"""
digraph GPTArch {
  rankdir=LR;
  splines=true;
  overlap=false;
  nodesep=0.6;
  ranksep=0.7;
  node [shape=box, style=rounded, fontname="Helvetica", color="#0A66C2"];
  edge [color="#555555"];

  subgraph cluster_embed {
    label="Embeddings";
    color="#cccccc";
    rank=same;
    tok [label="Token Embedding\n[V x D]"];
    pos [label="Position Embedding\n[T x D] or Sinusoidal"];
    add [label="Add\n[B, T, D]"];
    tok -> add;
    pos -> add;
  }

  subgraph cluster_stack {
    label="N x TransformerBlock (Pre‑Norm)";
    color="#cccccc";
    mha [label="Multi‑Head Attention\n[B, T, D] → [B, T, D]"];
    ffn [label="Feed‑Forward\n[B, T, D] → [B, T, D]"];
    mha -> ffn;
  }

  ln  [label="LayerNorm [B, T, D]"];
  head[label="LM Head (Linear)\n[D → V]"];

  add -> mha -> ffn -> ln -> head;
}
"""


def main() -> None:
    try:
        from graphviz import Source  # type: ignore

        s = Source(dot)
        out = s.render(filename=str(FIG_DIR / "ch09-gpt-arch"), format="svg", cleanup=True)
        print("Wrote:", out)
    except Exception as e:
        dot_path = FIG_DIR / "ch09-gpt-arch.dot"
        dot_path.write_text(dot)
        print("graphviz not available (", e, ")\nWrote DOT:", dot_path)
        print(
            "Render manually with:\n"
            "  dot -Tsvg figures/ch09-gpt-arch.dot -o figures/ch09-gpt-arch.svg"
        )


if __name__ == "__main__":
    main()
