"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Generate a toy BPE merges visualization as an SVG timeline.

Writes figures/appx-bpe-merges.svg without external deps.
"""

from __future__ import annotations


from pathlib import Path


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "appx-bpe-merges.svg"

    w, h = 1200, 260
    pad = 24
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        f'<text x="{w/2}" y="20" text-anchor="middle">Toy BPE merges over steps</text>',
    ]

    merges = [
        (1, '▁', 'model', '▁model'),
        (2, 'model', 's', 'models'),
        (3, '▁', 'token', '▁token'),
        (4, 'token', 's', 'tokens'),
        (5, '▁', 'learn', '▁learn'),
    ]

    x0, y0 = 60, 60
    step_gap = 36
    for i, (step, a, b, m) in enumerate(merges):
        y = y0 + i * step_gap
        items.append(f'<text x="{x0-30}" y="{y+6}" text-anchor="end">{step}</text>')
        # arrows a + b -> m
        # Left-hand symbols and operator spacing
        items.append(f'<text x="{x0}" y="{y+6}">{a}</text>')
        items.append(f'<text x="{x0+40}" y="{y+6}">+</text>')
        items.append(f'<text x="{x0+72}" y="{y+6}">{b}</text>')
        items.append(f'<text x="{x0+120}" y="{y+6}">→</text>')
        # Much wider merged rectangle to avoid overlaps and use available width
        rect_x = x0 + 150
        rect_w = 360
        items.append(f'<rect x="{rect_x}" y="{y-14}" width="{rect_w}" height="24" fill="#B5D0F5" stroke="#2b2b2b"/>')
        items.append(f'<text x="{rect_x + rect_w/2}" y="{y+4}" text-anchor="middle">{m}</text>')

    items.append('</svg>')
    out.write_text("\n".join(items))


if __name__ == '__main__':
    main()
