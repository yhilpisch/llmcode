"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Illustrate gradient-norm clipping with a synthetic curve.

Writes figures/ch13-clip.svg. No Matplotlib dependency required; generates a
simple SVG line for gradient norm and a horizontal clip threshold.
"""

from __future__ import annotations


from pathlib import Path
import math


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch13-clip.svg"

    w, h = 560, 220
    pad = 32
    steps = 200
    thr = 1.0
    xs = list(range(steps))
    # Synthetic noisy curve around 1.2 with spikes
    ys = [1.2 + 0.15 * math.sin(0.1 * i) + (0.0 if i % 37 else 1.2) for i in xs]

    def mapx(x): return pad + (w - 2*pad) * (x / (steps - 1))
    def mapy(y):
        ymin, ymax = 0.0, 2.8
        return h - pad - (h - 2*pad) * ((y - ymin) / (ymax - ymin))

    path = "M " + " ".join(f"{mapx(x):.1f},{mapy(y):.1f}" for x, y in zip(xs, ys))
    ythr = mapy(thr)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<text x="280" y="18" text-anchor="middle">Gradient norm with clipping</text>',
        f'<path d="{path}" fill="none" stroke="#0A66C2" stroke-width="2"/>',
        f'<line x1="{pad}" y1="{ythr:.1f}" x2="{w-pad}" y2="{ythr:.1f}" '
        'stroke="#DD4444" stroke-dasharray="4,3"/>',
        f'<text x="{w-pad}" y="{ythr-6:.1f}" text-anchor="end" fill="#DD4444">clip=1.0</text>',
        '</svg>'
    ]
    out.write_text("\n".join(svg))


if __name__ == '__main__':
    main()

