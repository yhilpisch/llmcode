"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Generate a sliding-window schematic for Chapter 10.

Writes figures/ch10-windows.svg. Uses Matplotlib if available; otherwise
falls back to a small hand-written SVG so the figure is always present.
"""

from __future__ import annotations


from pathlib import Path


def fallback_svg(out: Path, N: int = 24, T: int = 8) -> None:
    cell = 16
    pad = 18
    h = pad * 2 + cell * 3
    w = pad * 2 + cell * N
    y_ids = pad
    y_x = y_ids + cell
    y_y = y_x + cell
    # colors
    col_ids = "#DCE6F8"
    col_x = "#B5D0F5"
    col_y = "#9EC5F8"
    stroke = "#2b2b2b"
    style = (
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>'
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        style,
        f'<text x="{pad}" y="{y_ids-4}" fill="#222">ids</text>',
        f'<text x="{pad}" y="{y_x-4}" fill="#222">x = ids[i:i+T]</text>',
        f'<text x="{pad}" y="{y_y-4}" fill="#222">y = ids[i+1:i+T+1]</text>',
    ]
    # ids row
    for j in range(N):
        svg.append(
            f'<rect x="{pad + j*cell}" y="{y_ids}" width="{cell}" height="{cell}" '
            f'fill="{col_ids}" stroke="{stroke}" stroke-width="0.4" />'
        )
    # x window from j0..j0+T-1
    j0 = 4
    for j in range(T):
        xj = pad + (j0 + j) * cell
        svg.append(
            f'<rect x="{xj}" y="{y_x}" width="{cell}" height="{cell}" '
            f'fill="{col_x}" stroke="{stroke}" stroke-width="0.4" />'
        )
    # y window shifted by 1
    for j in range(T):
        xj = pad + (j0 + 1 + j) * cell
        svg.append(
            f'<rect x="{xj}" y="{y_y}" width="{cell}" height="{cell}" '
            f'fill="{col_y}" stroke="{stroke}" stroke-width="0.4" />'
        )
    svg.append('</svg>')
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch10-windows.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.style.use("seaborn-v0_8")
        N, T = 24, 8
        fig, ax = plt.subplots(figsize=(8.0, 1.8))
        ax.axis('off')
        y0 = 0
        # ids
        for j in range(N):
            ax.add_patch(plt.Rectangle((j, y0+0.8), 1, 0.8, fc="#DCE6F8", ec="#2b2b2b", lw=0.6))
        ax.text(-1.1, y0+1.3, 'ids', ha='right', va='center')
        # x
        j0 = 4
        for j in range(T):
            ax.add_patch(plt.Rectangle((j0+j, y0-0.2), 1, 0.8, fc="#B5D0F5", ec="#2b2b2b", lw=0.6))
        ax.text(-1.1, y0+0.2, 'x = ids[i:i+T]', ha='right', va='center')
        # y (shifted)
        for j in range(T):
            ax.add_patch(plt.Rectangle((j0+1+j, y0-1.2), 1, 0.8, fc="#9EC5F8", ec="#2b2b2b", lw=0.6))
        ax.text(-1.1, y0-0.8, 'y = ids[i+1:i+T+1]', ha='right', va='center')
        ax.set_xlim(-2, N+1); ax.set_ylim(-2, 3)
        fig.savefig(out, format='svg', bbox_inches='tight')
    except Exception:
        fallback_svg(out)


if __name__ == "__main__":
    main()
