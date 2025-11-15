"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Illustrate top-k and top-p filtering on a toy distribution.

Writes figures/ch11-topfilt.svg. Matplotlib if available, else fallback SVG.
"""

from __future__ import annotations


from pathlib import Path


def fallback_svg(out: Path) -> None:
    w, h = 600, 220
    pad = 28
    base = [0.40, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.03]
    cols = ["#0A66C2"] * len(base)
    def panel(x0, title, mask):
        svg = [f'<text x="{x0+140}" y="40" text-anchor="middle">{title}</text>']
        x = x0 + 16
        for p, m in zip(base, mask):
            height = (h - 80) * (p if not m else 0.02)
            color = "#0A66C2" if not m else "#DCE6F8"
            svg.append(
                f'<rect x="{x}" y="{h-30-height}" width="18" height="{height}" '
                f'fill="{color}" />'
            )
            x += 22
        return "\n".join(svg)
    topk_mask = [False, False, False, True, True, True, True, True]  # keep 3
    topp_mask = [False, False, False, False, True, True, True, True]  # keep to ~0.85
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        f'<text x="{w/2}" y="18" text-anchor="middle" fill="#222">Top-k vs Top-p</text>',
        panel(20, 'Top-k (k=3)', topk_mask),
        panel(320, 'Top-p (p≈0.85)', topp_mask),
        '</svg>'
    ]
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch11-topfilt.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-v0_8')
        base = np.array([0.40, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.03])
        topk_mask = np.array([False, False, False, True, True, True, True, True])
        topp_mask = np.array([False, False, False, False, True, True, True, True])
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), constrained_layout=True)
        axes[0].bar(range(len(base)), np.where(topk_mask, 0.02, base), color="#0A66C2")
        axes[0].set_title("Top-k (k=3)")
        axes[1].bar(range(len(base)), np.where(topp_mask, 0.02, base), color="#0A66C2")
        axes[1].set_title("Top-p (p≈0.85)")
        for ax in axes:
            ax.set_ylim(0, 0.5); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Top-k vs Top-p")
        fig.savefig(out, format='svg')
    except Exception:
        fallback_svg(out)


if __name__ == '__main__':
    main()

