"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Generate causal and combined (padding x causal) mask heatmaps for Ch. 9.

Always writes `figures/ch09-masks.svg`. Uses Matplotlib if available; otherwise
falls back to a minimal hand-written SVG so the book build never misses it.
"""

from __future__ import annotations


from pathlib import Path

import torch


def build_masks(T: int, pad_positions: list[int] | None = None):
    causal = torch.tril(torch.ones(T, T))  # [T, T]
    if not pad_positions:
        return causal, causal  # combined==causal in this trivial case
    pad = torch.ones(T)
    for p in pad_positions:
        if 0 <= p < T:
            pad[p] = 0
    pad_bt = pad[None, :]
    combined = pad_bt[:, None, :] * causal  # [1, T, T]
    return causal, combined.squeeze(0)


def render_svg_simple(causal: torch.Tensor, combined: torch.Tensor, out: Path) -> None:
    """Write a simple 2-panel SVG without external deps.

    Blue squares (1) vs white squares (0). Titles above each panel.
    """
    T = causal.size(0)
    cell = 16
    pad = 24
    gap = 40
    width = pad * 2 + cell * T * 2 + gap
    height = pad * 2 + cell * T + 28  # extra for titles
    def rects(mat: torch.Tensor, x0: int, y0: int) -> str:
        parts = []
        for i in range(T):          # rows (queries)
            for j in range(T):      # cols (keys)
                v = float(mat[i, j])
                color = "#0A66C2" if v > 0.5 else "#FFFFFF"
                parts.append(
                    f'<rect x="{x0 + j*cell}" y="{y0 + i*cell}" width="{cell}" '
                    f'height="{cell}" fill="{color}" stroke="#2b2b2b" stroke-width="0.4" />'
                )
        return "\n".join(parts)
    x1 = pad
    x2 = pad + cell * T + gap
    y = pad + 24
    style = (
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>'
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        style,
        f'<text x="{x1}" y="{pad}" fill="#222">Causal mask [T,T]</text>',
        f'<text x="{x2}" y="{pad}" fill="#222">Padding x causal [T,T]</text>',
        rects(causal, x1, y),
        rects(combined, x2, y),
        '</svg>',
    ]
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch09-masks.svg"
    causal, combined = build_masks(T=12, pad_positions=[9, 10, 11])
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
        im0 = axes[0].imshow(causal, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("Causal mask [T,T]")
        axes[0].set_xlabel("keys")
        axes[0].set_ylabel("queries")
        im1 = axes[1].imshow(combined, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Padding x causal [T,T]")
        axes[1].set_xlabel("keys")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(out, format="svg")
        print("Wrote:", out)
    except Exception:
        # Fallback: hand-written SVG so the book can include the figure
        render_svg_simple(causal, combined, out)
        print("Wrote (fallback SVG):", out)


if __name__ == "__main__":
    main()
