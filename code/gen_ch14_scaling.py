from __future__ import annotations

"""Generate a synthetic scaling law figure with axes and annotations.

Writes figures/ch14-scaling.svg (simple SVG; no external deps).
"""

from pathlib import Path
import math


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch14-scaling.svg"

    w, h = 680, 280
    pad = 46
    # Synthetic: loss = a * N^{-b} + c in log space (draw line with slight noise)
    xs = [10 ** (i / 10) for i in range(3, 33)]  # ~1e0..1e3
    a, b, c = 1.0, 0.3, 0.2
    ys = [a * (x ** (-b)) + c for x in xs]
    # Map to log10 for plotting
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    minx, maxx = min(lx), max(lx)
    miny, maxy = min(ly), max(ly)
    def mapx(x):
        return pad + (w - 2*pad) * ((x - minx) / (maxx - minx))
    def mapy(y):
        return h - pad - (h - 2*pad) * ((y - miny) / (maxy - miny))
    path = "M " + " ".join(
        f"{mapx(x):.1f},{mapy(y):.1f}" for x, y in zip(lx, ly)
    )

    # Horizontal line for irreducible error c (approx last y value)
    y_c = mapy(min(ly) + 0.02)

    # Slope annotation segment around the middle
    mid = len(lx) // 2
    x1, y1 = mapx(lx[mid] - 0.3), mapy(ly[mid] + 0.08)
    x2, y2 = mapx(lx[mid] + 0.3), mapy(ly[mid] - 0.08)

    # Axis label positions centered along their axes to avoid overlaps
    x_axis_mid_x = w / 2
    x_axis_label_y = h - pad + 24
    y_axis_mid_y = (h - pad + pad) / 2
    y_axis_label_x = pad - 34

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<defs>\n'
        '  <marker id="arrow" markerWidth="10" markerHeight="6" refX="10" refY="3" orient="auto">\n'
        '    <path d="M0,0 L10,3 L0,6 z" fill="#2b2b2b"/>\n'
        '  </marker>\n'
        '</defs>',
        # Title
        f'<text x="{w/2:.1f}" y="20" text-anchor="middle">Scaling law: log loss vs log scale</text>',
        # Axes
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#2b2b2b" marker-end="url(#arrow)"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{pad}" y2="{pad}" stroke="#2b2b2b" marker-end="url(#arrow)"/>',
        # Centered axis labels
        f'<text x="{x_axis_mid_x:.1f}" y="{x_axis_label_y:.1f}" text-anchor="middle">log10(N)</text>',
        f'<text x="{y_axis_label_x:.1f}" y="{y_axis_mid_y:.1f}" text-anchor="middle" transform="rotate(-90 {y_axis_label_x:.1f},{y_axis_mid_y:.1f})">log10(loss)</text>',
        # qualitative end labels
        f'<text x="{pad+6}" y="{h-pad+16}" fill="#555">small</text>',
        f'<text x="{w-pad-40}" y="{h-pad+16}" fill="#555">large</text>',
        f'<text x="{pad-14}" y="{h-pad-6}" fill="#555" text-anchor="end">low</text>',
        f'<text x="{pad-14}" y="{pad+6}" fill="#555" text-anchor="end">high</text>',
        # Curve
        f'<path d="{path}" fill="none" stroke="#0A66C2" stroke-width="2"/>',
        # Irreducible error line c with centered annotation
        f'<line x1="{pad}" y1="{y_c:.1f}" x2="{w-pad}" y2="{y_c:.1f}" stroke="#DD4444" stroke-dasharray="5,4"/>',
        f'<text x="{w/2:.1f}" y="{y_c-8:.1f}" text-anchor="middle" fill="#DD4444">irreducible error c</text>',
        # Slope annotation
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#555555"/>',
        f'<text x="{(x1+x2)/2+8:.1f}" y="{(y1+y2)/2-8:.1f}" fill="#555">slope ≈ −b</text>',
        '</svg>'
    ]
    out.write_text("\n".join(svg))


if __name__ == '__main__':
    main()
