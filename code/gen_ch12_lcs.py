from __future__ import annotations

"""Draw an LCS alignment sketch for ROUGE-L intuition.

Writes figures/ch12-lcs.svg with two token rows and highlighted matches.
Falls back to simple SVG so the book always builds.
"""

from pathlib import Path


def fallback_svg(out: Path) -> None:
    w, h = 680, 180
    pad = 24
    cell = 18
    hyp = ["the", "cat", "sat", "on", "the", "mat"]
    ref = ["the", "cat", "is", "on", "the", "mat"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<text x="20" y="20">ROUGE-L via LCS</text>'
    ]
    # token rows
    x0 = pad + 120
    for i, tok in enumerate(ref):
        x = x0 + i * (cell + 10)
        svg.append(f'<rect x="{x}" y="40" width="{cell}" height="{cell}" fill="#DCE6F8" stroke="#2b2b2b"/>' )
        svg.append(f'<text x="{x+cell/2}" y="{40+cell+14}" text-anchor="middle">{tok}</text>')
    for i, tok in enumerate(hyp):
        x = x0 + i * (cell + 10)
        svg.append(f'<rect x="{x}" y="90" width="{cell}" height="{cell}" fill="#B5D0F5" stroke="#2b2b2b"/>' )
        svg.append(f'<text x="{x+cell/2}" y="{90+cell+14}" text-anchor="middle">{tok}</text>')
    # highlight LCS edges (the, cat, on, the, mat)
    match_idx = [(0,0),(1,1),(3,3),(4,4),(5,5)]
    for hi, ri in match_idx:
        xh = x0 + hi * (cell + 10) + cell/2
        xr = x0 + ri * (cell + 10) + cell/2
        svg.append(f'<line x1="{xr}" y1="40" x2="{xh}" y2="{90+cell}" stroke="#0A66C2"/>')
    svg.append('</svg>')
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch12-lcs.svg"
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        # fallback is sufficient visually; keep matplotlib path minimal
        fallback_svg(out)
    except Exception:
        fallback_svg(out)


if __name__ == '__main__':
    main()

