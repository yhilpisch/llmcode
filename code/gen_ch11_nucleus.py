from __future__ import annotations

"""Plot cumulative probability and nucleus threshold p.

Writes figures/ch11-nucleus.svg. Uses Matplotlib if available; otherwise
falls back to a minimal SVG line/area plot to ensure the book builds.
"""

from pathlib import Path


def fallback_svg(out: Path) -> None:
    w, h = 540, 220
    pad = 32
    # Toy sorted probabilities
    probs = [0.4, 0.25, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02]
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)
    pthr = 0.9
    # Map to svg coords
    def mapx(i: int) -> float:
        return pad + (w - 2 * pad) * (i / (len(cum) - 1))
    def mapy(y: float) -> float:
        return h - pad - (h - 2 * pad) * y
    path = "M " + " ".join(f"{mapx(i):.1f},{mapy(y):.1f}" for i, y in enumerate(cum))
    ythr = mapy(pthr)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        f'<text x="{w/2}" y="18" text-anchor="middle" fill="#222">Nucleus threshold</text>',
        f'<path d="{path}" fill="none" stroke="#0A66C2" stroke-width="2" />',
        f'<line x1="{pad}" y1="{ythr:.1f}" x2="{w-pad}" y2="{ythr:.1f}" '
        f'stroke="#DD4444" stroke-dasharray="4,3" />',
        f'<text x="{w-pad}" y="{ythr-6:.1f}" text-anchor="end" fill="#DD4444">p=0.9</text>',
        '</svg>'
    ]
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch11-nucleus.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-v0_8')
        probs = np.array([0.4, 0.25, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02])
        cum = np.cumsum(probs)
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.plot(cum, color="#0A66C2", lw=2)
        p = 0.9
        ax.axhline(p, color="#DD4444", ls='--')
        ax.text(len(cum)-1, p + 0.03, f"p={p}", color="#DD4444",
                ha='right', va='bottom')
        ax.set_xlim(0, len(cum)-1)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Nucleus threshold")
        fig.tight_layout()
        fig.savefig(out, format='svg')
    except Exception:
        fallback_svg(out)


if __name__ == '__main__':
    main()

