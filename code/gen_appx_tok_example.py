"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Generate a simple SVG illustrating char/word/BPE tokenization on one sentence.

Writes figures/appx-tok-example.svg using dependency-free SVG drawing.
"""

from __future__ import annotations


from pathlib import Path


def draw_row(x0, y0, tokens, color="#B5D0F5", gap=8, pad=6, h=34):
    items = []
    x = x0
    for t in tokens:
        w = max(40, 9 * len(t))
        items.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" fill="{color}" stroke="#2b2b2b"/>')
        items.append(f'<text x="{x + w/2}" y="{y0 + h/2 + 4}" text-anchor="middle">{t}</text>')
        x += w + gap
    return items


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "appx-tok-example.svg"

    sent = "The model models tokens"
    char = list(sent)
    word = sent.split()
    # pseudo-BPE pieces for illustration
    bpe = ["The", "\u2581model", "\u2581model", "s", "\u2581token", "s"]

    w, h = 760, 240
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<style> text{font-family:Helvetica,Arial,sans-serif;font-size:12px;}</style>',
        f'<text x="{w/2}" y="20" text-anchor="middle">Tokenization variants</text>',
        '<text x="16" y="56">Character</text>',
        '<text x="16" y="116">Word</text>',
        '<text x="16" y="176">BPE (toy)</text>',
    ]
    items += draw_row(100, 36, char, color="#DCE6F8", h=28)
    items += draw_row(100, 96, word, color="#CFE2FF")
    items += draw_row(100, 156, bpe, color="#B5D0F5")
    items.append('</svg>')
    out.write_text("\n".join(items))


if __name__ == '__main__':
    main()

