from __future__ import annotations

"""Visualize the effect of temperature on a toy logit vector.

Writes figures/ch11-temp.svg. Matplotlib if available, else fallback SVG.
"""

from pathlib import Path


def fallback_svg(out: Path) -> None:
    w, h = 520, 220
    pad = 28
    bars = [0.55, 0.25, 0.1, 0.06, 0.04]
    cols = ["#0A66C2", "#5491D6", "#7FADE5", "#A5C5EE", "#C9DCF7"]
    def bar(x, y, w_, h_, c):
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w_:.1f}" height="{h_:.1f}" '
                f'fill="{c}" />')
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        f'<text x="{w/2}" y="18" text-anchor="middle" fill="#222">Temperature</text>'
    ]
    # draw three panels (T=0.7, 1.0, 1.3) with simple bars
    panel_w = (w - 2*pad) / 3
    for i, t in enumerate([0.7, 1.0, 1.3]):
        x0 = pad + i * panel_w
        svg.append(f'<text x="{x0+panel_w/2}" y="40" text-anchor="middle">T={t}</text>')
        maxh = h - 80
        for j, p in enumerate(bars):
            height = maxh * (p ** (1.0 if t==1.0 else (1.2 if t<1 else 0.8)))
            svg.append(bar(x0 + 16 + j*24, h-30-height, 18, height, cols[j]))
    svg.append('</svg>')
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch11-temp.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-v0_8')
        logits = np.array([2.0, 1.0, 0.0, -0.5, -1.0])
        Ts = [0.7, 1.0, 1.3]
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2), constrained_layout=True)
        for ax, T in zip(axes, Ts):
            p = np.exp(logits / T); p = p / p.sum()
            ax.bar(range(len(p)), p, color="#0A66C2")
            ax.set_title(f"T={T}")
            ax.set_ylim(0, 1.0)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Temperature")
        fig.savefig(out, format='svg')
    except Exception:
        fallback_svg(out)


if __name__ == '__main__':
    main()

