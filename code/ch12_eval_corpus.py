"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Quick corpus evaluator for Chapter 12.

Reads references and hypotheses from files (one example per line). The
references file supports multiple references per example by separating them
with the delimiter " ||| ". Tokenization defaults to whitespace with optional
lowercasing.

Outputs BLEU (corpus), ROUGE-L, METEOR (simplified), and distinct-1/2.
"""

from __future__ import annotations


import argparse
from pathlib import Path
from typing import List, Sequence

from code.ch12_metrics_text import (
    bleu_corpus,
    rouge_l,
    meteor_simple,
    distinct_n,
)


def parse_lines(path: str, lowercase: bool) -> List[str]:
    text = Path(path).read_text(encoding="utf-8").splitlines()
    return [t.lower() if lowercase else t for t in text]


def to_refs(lines: List[str]) -> List[List[Sequence[str]]]:
    """Split each line on ' ||| ' to allow multiple references per example."""
    out: List[List[Sequence[str]]] = []
    for line in lines:
        refs = [seg.strip().split() for seg in line.split(" ||| ")]
        out.append(refs)
    return out


def to_hyps(lines: List[str]) -> List[Sequence[str]]:
    return [ln.split() for ln in lines]


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate text outputs against references")
    p.add_argument("--refs", required=True, help="path to references.txt")
    p.add_argument("--hyps", required=True, help="path to hypotheses.txt")
    p.add_argument("--lower", action="store_true", help="lowercase before tokenizing")
    p.add_argument("--max-n", type=int, default=4, help="max n-gram for BLEU")
    args = p.parse_args()

    ref_lines = parse_lines(args.refs, args.lower)
    hyp_lines = parse_lines(args.hyps, args.lower)
    if len(ref_lines) != len(hyp_lines):
        raise SystemExit("refs and hyps must have the same number of lines")

    references = to_refs(ref_lines)
    hypotheses = to_hyps(hyp_lines)

    bleu = bleu_corpus(references, hypotheses, max_n=args.max_n, smooth=True)
    rlg = rouge_l(references, hypotheses)
    met = meteor_simple(references, hypotheses)
    d1 = distinct_n(hypotheses, 1)
    d2 = distinct_n(hypotheses, 2)

    print("Examples:", len(hypotheses))
    print(f"BLEU_{args.max_n}:  {bleu:.3f}")
    print(f"ROUGE_L:  {rlg:.3f}")
    print(f"METEOR*: {met:.3f}  (simplified)")
    print(f"distinct-1: {d1:.3f}")
    print(f"distinct-2: {d2:.3f}")


if __name__ == "__main__":
    main()

