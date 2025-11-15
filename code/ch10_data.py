"""
Building a Large Language Model from Scratch
— A Step-by-Step Guide Using Python and PyTorch

(c) Dr. Yves J. Hilpisch (The Python Quants GmbH)
AI-Powered by GPT-5.

Small data helpers for Chapter 10: building a token id stream and slicing
into (input, target) chunks for next-token prediction.

We keep this self-contained and friendly:
- A minimal dataset that returns (x, y) where y is x shifted by one.
- Helpers to build ids from raw text using either a provided tokenizer or a
  byte-level fallback (0-255).
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class TextIds:
    """Container for a 1-D id stream and tokenizer metadata.

    - ids: concatenated token ids (1-D)
    - vocab_size: size of the token vocabulary
    - pad_id: optional pad index (for CE ignore_index)
    - unk_id: optional unknown-token index
    - level: optional tokenization level ('byte'|'char'|'word')
    - id_to_token: optional list of tokens by index for reconstruction/decoding
    """
    ids: torch.Tensor
    vocab_size: int
    pad_id: int | None = None
    unk_id: int | None = None
    level: str | None = None
    id_to_token: list[str] | None = None


def load_texts(paths: Sequence[str] | None) -> str:
    if not paths:
        return "Hello world. Hello vectors.\n"
    texts: List[str] = []
    for p in paths:
        # Read each file as UTF-8 and concatenate with newlines
        data = Path(p).read_text(encoding="utf-8")
        texts.append(data)
    return "\n".join(texts)


def build_ids_byte_level(text: str) -> TextIds:
    # Encode to bytes and map each byte directly to an id
    data = text.encode("utf-8", errors="ignore")
    ids = torch.tensor(list(data), dtype=torch.long)
    return TextIds(
        ids=ids,
        vocab_size=256,
        pad_id=None,
        unk_id=None,
        level="byte",
        id_to_token=None,
    )


def build_ids_with_tokenizer(text: str, level: str = "char") -> TextIds:
    """Use the SimpleTokenizer from Chapter 6 if available; else byte-level.

    We try local imports first so running `python code/...` works without
    installing `code/` as a package.
    """
    try:
        # when executing scripts under code/, neighbors are importable
        from ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore
    except Exception:
        try:
            from code.ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore
        except Exception:
            return build_ids_byte_level(text)
    if hasattr(SimpleTokenizer, "from_text"):
        tok = SimpleTokenizer.from_text(  # type: ignore[attr-defined]
            text, level=level
        )
    else:
        # Build from raw text using the module's helpers
        tokens = SimpleTokenizer._split(text, level)
        vocab = Vocab.build(tokens)
        tok = SimpleTokenizer(vocab=vocab, level=level)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return TextIds(
        ids=ids,
        vocab_size=len(tok.vocab),
        pad_id=tok.pad,
        unk_id=tok.unk,
        level=level,
        id_to_token=list(tok.vocab.id_to_token),
    )


class LMSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Slice a long id stream into overlapping (x,y) chunks of length T.

    x is ids[i : i+T], y is ids[i+1 : i+T+1]. The number of samples is
    len(ids) - T.
    """

    def __init__(self, ids: torch.Tensor, block_size: int):
        assert ids.ndim == 1 and ids.dtype == torch.long
        self.ids = ids
        self.T = int(block_size)

    def __len__(self) -> int:
        return max(0, self.ids.numel() - self.T)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        x = self.ids[i : i + self.T]
        y = self.ids[i + 1 : i + self.T + 1]
        return x, y


__all__ = [
    "TextIds",
    "load_texts",
    "build_ids_byte_level",
    "build_ids_with_tokenizer",
    "LMSequenceDataset",
]
