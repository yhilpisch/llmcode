"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-powered by GPT-5.x.

Educational text metrics for Chapter 12.

This module implements small, dependency‑free versions of common metrics:

- BLEU (corpus): n‑gram precision with brevity penalty, optional Add‑1 smoothing.
- ROUGE‑L: F‑measure based on the longest common subsequence (LCS).
- METEOR (simplified): unigram precision/recall F‑mean with a fragmentation
  penalty estimated from contiguous matching chunks (no stemming/synonyms).
- Diversity helpers: distinct‑1 / distinct‑2.

Inputs are tokenized sequences (lists of strings or ints). We keep the
implementations compact and readable for teaching; they are not drop‑in
replacements for official packages, but align with the main ideas.
"""

from __future__ import annotations


from typing import Iterable, List, Sequence, Tuple
from collections import Counter


def _ngram_counts(tokens: Sequence, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_corpus(
    references: List[List[Sequence]],
    hypotheses: List[Sequence],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Compute a simple corpus BLEU.

    Args:
        references: list of lists of reference token sequences (per hypothesis)
        hypotheses: list of hypothesis token sequences
        max_n: highest n‑gram order (default 4)
        smooth: Add‑1 smoothing for counts
    Returns:
        BLEU score in [0, 1]
    """
    assert len(references) == len(hypotheses)

    # Modified n‑gram precisions
    num = [0] * max_n
    den = [0] * max_n

    ref_len = 0
    hyp_len = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_len += len(hyp)
        # reference length closest to hypothesis (brevity penalty)
        ref_lengths = [len(r) for r in refs]
        ref_len += min(ref_lengths, key=lambda rl: (abs(rl - len(hyp)), rl))

        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            max_ref_counts: Counter = Counter()
            for r in refs:
                max_ref_counts |= _ngram_counts(r, n)
            # clipped counts
            overlap = {
                g: min(c, max_ref_counts.get(g, 0)) for g, c in hyp_counts.items()
            }
            num[n - 1] += sum(overlap.values())
            den[n - 1] += max(1, sum(hyp_counts.values()))

    # Smoothed precisions
    precisions = []
    for i in range(max_n):
        if smooth:
            precisions.append((num[i] + 1) / (den[i] + 1))
        else:
            precisions.append(0.0 if den[i] == 0 else num[i] / den[i])

    # Brevity penalty
    import math

    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(1, hyp_len))

    # Geometric mean of precisions
    gm = math.exp(sum((1 / max_n) * math.log(max(p, 1e-16)) for p in precisions))
    return bp * gm


def _lcs_length(a: Sequence, b: Sequence) -> int:
    # Classic DP for LCS length (O(len(a)*len(b)))
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]


def rouge_l(
    references: List[List[Sequence]],
    hypotheses: List[Sequence],
    beta: float = 1.2,
) -> float:
    """Compute ROUGE‑L F‑measure averaged across examples.

    For each hypothesis we take the best reference by LCS F‑measure.
    """
    import math

    scores: List[float] = []
    for refs, hyp in zip(references, hypotheses):
        best = 0.0
        for r in refs:
            lcs = _lcs_length(r, hyp)
            if lcs == 0:
                continue
            prec = lcs / max(1, len(hyp))
            rec = lcs / max(1, len(r))
            if prec == 0 and rec == 0:
                f = 0.0
            else:
                beta2 = beta * beta
                f = (1 + beta2) * prec * rec / max(beta2 * prec + rec, 1e-12)
            best = max(best, f)
        scores.append(best)
    return sum(scores) / max(1, len(scores))


def _matching_chunks(h: Sequence, r: Sequence) -> Tuple[int, int]:
    """Return (matches, chunks) for contiguous exact matches between h and r.

    Used for a simplified METEOR chunk penalty.
    """
    # Build index of tokens in r
    from collections import defaultdict

    pos = defaultdict(list)
    for j, tok in enumerate(r):
        pos[tok].append(j)

    matches = 0
    chunks = 0
    prev_j = None
    for tok in h:
        if not pos[tok]:
            continue
        j = pos[tok].pop(0)  # greedy match leftmost
        matches += 1
        if prev_j is None or j != prev_j + 1:
            chunks += 1
        prev_j = j
    return matches, chunks


def meteor_simple(
    references: List[List[Sequence]],
    hypotheses: List[Sequence],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """Simplified METEOR.

    For each (refs, hyp):
    - Compute unigram precision P and recall R against each ref (exact match).
    - F_mean = (P*R) / (alpha*P + (1 - alpha)*R)
    - Compute chunk penalty: Pen = gamma * (chunks / matches) ** beta
    - Score = F_mean * (1 - Pen)
    We take the max score over references and average over the corpus.
    """
    import math

    scores: List[float] = []
    for refs, hyp in zip(references, hypotheses):
        best = 0.0
        for r in refs:
            # unigram matches
            hyp_counts = Counter(hyp)
            ref_counts = Counter(r)
            overlap = sum(min(hyp_counts[t], ref_counts[t]) for t in hyp_counts)
            P = overlap / max(1, len(hyp))
            R = overlap / max(1, len(r))
            if P == 0 or R == 0:
                cand = 0.0
            else:
                Fm = (P * R) / max(alpha * P + (1 - alpha) * R, 1e-12)
                m, ch = _matching_chunks(hyp, r)
                if m == 0:
                    penalty = 0.0
                else:
                    penalty = gamma * ((ch / m) ** beta)
                cand = Fm * (1 - penalty)
            best = max(best, cand)
        scores.append(best)
    return sum(scores) / max(1, len(scores))


def distinct_n(hypotheses: List[Sequence], n: int = 1) -> float:
    """Proportion of distinct n‑grams across all hypotheses (diversity)."""
    grams = Counter()
    total = 0
    for h in hypotheses:
        c = _ngram_counts(h, n)
        grams.update(c)
        total += sum(c.values())
    if total == 0:
        return 0.0
    return len(grams) / total


__all__ = [
    "bleu_corpus",
    "rouge_l",
    "meteor_simple",
    "distinct_n",
]

