from __future__ import annotations

"""Generate a simple LR warmup curve figure for Chapter 10.

Writes figures/ch10-lr-warmup.svg. Uses Matplotlib if available; otherwise
falls back to a small hand-written SVG path so the figure is always present.
"""

from pathlib import Path


def fallback_svg(out: Path, steps: int = 200, warmup: int = 50) -> None:
    w, h = 460, 180
    pad = 32
    # Build points for linear warmup to 1 and then flat
    xs = list(range(steps))
    ys = [(x+1)/warmup if x < warmup else 1.0 for x in xs]
    # map to svg coords
    def mapx(x):
        return pad + (w - 2*pad) * (x / max(1, steps-1))
    def mapy(y):
        # y in [0,1] -> svg y downwards
        return h - pad - (h - 2*pad) * y
    path = "M " + " ".join(f"{mapx(x):.1f},{mapy(y):.1f}" for x, y in zip(xs, ys))
    style = (
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>'
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        style,
        f'<text x="{w/2}" y="16" text-anchor="middle" fill="#222">LR warmup</text>',
        f'<path d="{path}" fill="none" stroke="#0A66C2" stroke-width="2" />',
        '</svg>'
    ]
    out.write_text("\n".join(svg))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch10-lr-warmup.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use("seaborn-v0_8")
        steps, warmup = 200, 50
        xs = np.arange(steps)
        ys = np.minimum(1.0, (xs + 1) / float(warmup))
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.plot(xs, ys, color="#0A66C2")
        ax.set_title("LR warmup")
        ax.set_xlabel("step")
        ax.set_ylabel("scale")
        fig.tight_layout()
        fig.savefig(out, format='svg')
    except Exception:
        fallback_svg(out)


if __name__ == "__main__":
    main()
