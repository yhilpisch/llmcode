from __future__ import annotations

"""Illustrate gradient accumulation: k micro-batches per optimizer step.

Writes figures/ch13-accum.svg as a simple, dependency-free SVG.
"""

from pathlib import Path


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch13-accum.svg"

    w, h = 700, 220
    pad = 24
    cell_w, cell_h = 110, 44
    gap = 18

    def batch(x, y, label, color="#B5D0F5"):
        return [
            f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
            f'fill="{color}" stroke="#2b2b2b"/>',
            f'<text x="{x+cell_w/2}" y="{y+cell_h/2+4}" text-anchor="middle">{label}</text>',
        ]

    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w, h),
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<text x="350" y="20" text-anchor="middle">Gradient accumulation (k micro-batches per step)</text>',
    ]

    x0 = pad + 20
    y0 = 60
    k = 4
    for i in range(k):
        x = x0 + i * (cell_w + gap)
        items += batch(x, y0, f"micro-batch {i+1}")
        # plus sign between micro-batches
        if i < k - 1:
            items.append(f'<text x="{x+cell_w+gap/2}" y="{y0+cell_h/2+4}" text-anchor="middle">+</text>')

    # Arrow to optimizer step box
    x_end = x0 + (k-1) * (cell_w + gap) + cell_w + 40
    y_mid = y0 + cell_h/2
    items.append(f'<line x1="{x0+ k*(cell_w+gap) - gap}" y1="{y_mid}" x2="{x_end}" y2="{y_mid}" stroke="#0A66C2" marker-end="url(#arrow)"/>')

    # Optimizer step box
    step_x, step_y = x_end, y0
    items += batch(step_x, step_y, "optimizer step", color="#9EC5F8")

    # Define arrow marker
    items.insert(1, (
        '<defs><marker id="arrow" markerWidth="10" markerHeight="6" refX="10" refY="3" orient="auto">'
        '<path d="M0,0 L10,3 L0,6 z" fill="#0A66C2"/></marker></defs>'
    ))

    items.append('</svg>')
    out.write_text("\n".join(items))


if __name__ == "__main__":
    main()

