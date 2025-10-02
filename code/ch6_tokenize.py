from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad: int
    unk: int

    @classmethod
    def build(
        cls,
        tokens: Iterable[str],
        min_freq: int = 1,
        specials: Iterable[str] = ("<PAD>", "<UNK>"),
    ) -> "Vocab":
        counter = Counter(tokens)
        id_to_token = list(specials)
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in id_to_token:
                id_to_token.append(tok)
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        pad = token_to_id[specials[0]]
        unk = token_to_id[specials[1]]
        return cls(token_to_id, id_to_token, pad, unk)

    def __len__(self) -> int:
        return len(self.id_to_token)


class SimpleTokenizer:
    """Tiny tokenizer for chapter 6 (char or word level)."""

    def __init__(self, vocab: Vocab, level: str = "char") -> None:
        assert level in {"char", "word"}
        self.vocab = vocab
        self.level = level
        self.pad = vocab.pad
        self.unk = vocab.unk

    @staticmethod
    def _split(text: str, level: str) -> List[str]:
        if level == "char":
            return list(text)
        # simple whitespace/punct split for demo purposes
        out: List[str] = []
        token = []
        for ch in text:
            if ch.isalnum():
                token.append(ch.lower())
            else:
                if token:
                    out.append("".join(token))
                    token = []
                if ch.strip():  # keep punctuation as its own token
                    out.append(ch)
        if token:
            out.append("".join(token))
        return out

    @classmethod
    def from_file(cls, path: str | Path, level: str = "char", min_freq: int = 1) -> "SimpleTokenizer":
        text = Path(path).read_text(encoding="utf-8")
        tokens = cls._split(text, level)
        vocab = Vocab.build(tokens, min_freq=min_freq)
        return cls(vocab=vocab, level=level)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in self._split(text, self.level):
            ids.append(self.vocab.token_to_id.get(tok, self.unk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks: List[str] = []
        for i in ids:
            if 0 <= i < len(self.vocab.id_to_token):
                tok = self.vocab.id_to_token[i]
                if tok not in {"<PAD>", "<UNK>"}:
                    toks.append(tok)
            else:
                toks.append("<UNK>")
        if self.level == "char":
            return "".join(toks)
        # naive word join: put space before alphanumerics only
        out: List[str] = []
        for t in toks:
            if not out:
                out.append(t)
            elif t.isalnum():
                out.append(" " + t)
            else:
                out.append(t)
        return "".join(out)
