"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Generate a simple LLM build roadmap diagram.

If `graphviz` Python package is installed, renders SVG directly.
Else, writes `figures/llm-roadmap.dot` for manual rendering:

    dot -Tsvg figures/llm-roadmap.dot -o figures/llm-roadmap.svg
"""

from __future__ import annotations

from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

dot_content = r"""
digraph LLMRoadmap {
  rankdir=LR;
  node [shape=box, style=rounded, color="#0A66C2", fontname="Helvetica"];
  edge [color="#555555"];

  setup     [label="Repo Setup\n& Env Checks"];
  data      [label="Data\n& Tokenization"];
  model     [label="Embeddings\n+ Transformer Blocks"];
  training  [label="Training\n(CE Loss, AdamW)"];
  sampling  [label="Sampling\n(top-k, top-p)"];
  eval      [label="Evaluation\n(Perplexity & More)"];
  deploy    [label="Deployment\n(CLI, App, API)"];

  setup -> data -> model -> training -> sampling -> eval -> deploy;
}
"""

def main() -> None:
    try:
        from graphviz import Source  # type: ignore

        s = Source(dot_content)
        out = s.render(filename=str(FIG_DIR / "llm-roadmap"), format="svg", cleanup=True)
        print("Wrote:", out)
    except Exception as e:
        dot_path = FIG_DIR / "llm-roadmap.dot"
        dot_path.write_text(dot_content)
        print("graphviz not available (", e, ")\nWrote DOT:", dot_path)
        print(
            "Render manually with:\n"
            "  dot -Tsvg figures/llm-roadmap.dot -o figures/llm-roadmap.svg"
        )


if __name__ == "__main__":
    main()
