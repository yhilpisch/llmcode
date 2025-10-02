from __future__ import annotations

"""Draw warmup + cosine LR schedule used in Chapter 13.

Writes figures/ch13-lr-cosine.svg. Falls back to minimal SVG if Matplotlib
is unavailable.
"""

from pathlib import Path


def fallback_svg(out: Path) -> None:
    w, h = 560, 220
    pad = 32
    warmup, total, minr = 100, 1000, 0.1
    xs = list(range(total))
    ys = []
    import math
    for s in xs:
        s1 = s + 1
        if s1 <= warmup:
            ys.append(s1 / warmup)
        else:
            t = s1 - warmup
            frac = t / (total - warmup)
            cos = 0.5 * (1 + math.cos(math.pi * frac))
            ys.append(minr + (1 - minr) * cos)
    def mapx(x): return pad + (w - 2*pad) * (x / (total-1))
    def mapy(y): return h - pad - (h - 2*pad) * y
    path = "M " + " ".join(f"{mapx(x):.1f},{mapy(y):.1f}" for x,y in zip(xs,ys))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<text x="280" y="18" text-anchor="middle">Warmup + Cosine LR</text>',
        f'<path d="{path}" fill="none" stroke="#0A66C2" stroke-width="2"/>',
        '</svg>'
    ]
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch13-lr-cosine.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-v0_8')
        warmup, total, minr = 100, 1000, 0.1
        xs = np.arange(total)
        ys = []
        for s in xs:
            s1 = s + 1
            if s1 <= warmup:
                ys.append(s1 / warmup)
            else:
                t = s1 - warmup
                frac = t / (total - warmup)
                ys.append(minr + (1 - minr) * 0.5 * (1 + np.cos(np.pi * frac)))
        ys = np.array(ys)
        fig, ax = plt.subplots(figsize=(6.4, 2.2))
        ax.plot(xs, ys, color="#0A66C2")
        ax.set_title("Warmup + Cosine LR")
        ax.set_xlabel("step"); ax.set_ylabel("scale")
        fig.tight_layout(); fig.savefig(out, format='svg')
    except Exception:
        fallback_svg(out)


if __name__ == '__main__':
    main()

