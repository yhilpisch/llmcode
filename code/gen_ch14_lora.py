from __future__ import annotations

"""Generate a simple LoRA diagram: base Linear plus low-rank delta.

Writes figures/ch14-lora.svg without external dependencies.
"""

from pathlib import Path


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "ch14-lora.svg"

    w, h = 760, 260
    pad = 30
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        '<defs>\n'
        '  <marker id="arrow" markerWidth="10" markerHeight="6"\n'
        '          refX="10" refY="3" orient="auto">\n'
        '    <path d="M0,0 L10,3 L0,6 z" fill="#0A66C2"/>\n'
        '  </marker>\n'
        '</defs>',
    ]
    # Base linear block (center)
    x0, y0 = pad + 80, 100
    items.append(
        f'<rect x="{x0}" y="{y0}" width="200" height="60" fill="#DCE6F8" stroke="#2b2b2b"/>'
    )
    items.append(
        f'<text x="{x0+100}" y="{y0+36}" text-anchor="middle">Base Linear W</text>'
    )
    # A and B blocks (adapter branch)
    ax, ay = x0 + 250, y0 - 50
    items.append(
        f'<rect x="{ax}" y="{ay}" width="140" height="34" fill="#B5D0F5" stroke="#2b2b2b"/>'
    )
    items.append(
        f'<text x="{ax+70}" y="{ay+22}" text-anchor="middle">A (r × d_in)</text>'
    )
    bx, by = ax, y0 + 76
    items.append(
        f'<rect x="{bx}" y="{by}" width="140" height="34" fill="#B5D0F5" stroke="#2b2b2b"/>'
    )
    items.append(
        f'<text x="{bx+70}" y="{by+22}" text-anchor="middle">B (d_out × r)</text>'
    )
    # Input x arrow into base and into A (branch)
    items.append(
        f'<line x1="{x0-40}" y1="{y0+30}" x2="{x0}" y2="{y0+30}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    items.append(f'<text x="{x0-44}" y="{y0+34}" text-anchor="end">x</text>')
    items.append(
        f'<line x1="{x0-40}" y1="{ay+17}" x2="{ax}" y2="{ay+17}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    # A to B, B to sum
    items.append(
        f'<line x1="{ax+140}" y1="{ay+17}" x2="{bx+70}" y2="{by}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    sumx, sumy = x0 + 470, y0 + 30
    items.append(f'<circle cx="{sumx}" cy="{sumy}" r="10" fill="#FFFFFF" stroke="#2b2b2b"/>')
    items.append(f'<text x="{sumx}" y="{sumy+4}" text-anchor="middle">+</text>')
    items.append(
        f'<line x1="{x0+200}" y1="{y0+30}" x2="{sumx-10}" y2="{sumy}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    items.append(
        f'<line x1="{bx+140}" y1="{by+17}" x2="{sumx-10}" y2="{sumy}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    # Scale label α/r on the adapter path
    items.append(f'<text x="{bx+118}" y="{by-8}">scale: α/r</text>')
    # Sum to output
    outx = sumx + 160
    items.append(
        f'<line x1="{sumx+10}" y1="{sumy}" x2="{outx}" y2="{sumy}" stroke="#0A66C2" marker-end="url(#arrow)"/>'
    )
    items.append(f'<text x="{outx+10}" y="{sumy+4}">output</text>')
    # Annotations
    items.append(
        f'<text x="{x0}" y="{y0-10}">x</text>'
    )
    items.append(f'<text x="{ax+160}" y="{ay+22}">ΔW = B @ A</text>')
    items.append('</svg>')
    out.write_text("\n".join(items))


if __name__ == '__main__':
    main()
