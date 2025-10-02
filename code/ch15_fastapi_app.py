from __future__ import annotations

"""FastAPI app serving a minimal /generate endpoint (Chapter 15)."""

import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Import from code/
sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch11_sampling import sample  # type: ignore
from ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore


class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 0
    top_p: float = 0.95


def load_bundle(path: str):
    b = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**b["config"])  # type: ignore
    model = GPT(cfg).eval()
    model.load_state_dict(b["model_state"])  # type: ignore
    meta = b.get("tokenizer")
    tok = None
    if meta and meta.get("id_to_token"):
        id_to_token = list(meta["id_to_token"])  # ensure list
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        vocab = Vocab(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad=int(meta.get("pad_id", 0)),
            unk=int(meta.get("unk_id", 1)),
        )
        tok = SimpleTokenizer(vocab=vocab, level=meta.get("level", "char"))
    return model, tok


app = FastAPI(title="Mini‑GPT")
MODEL, TOK = None, None


@app.on_event("startup")
def _startup():
    global MODEL, TOK
    bundle = Path("model_bundle.pt")
    if bundle.exists():
        MODEL, TOK = load_bundle(str(bundle))


@app.post("/generate")
def generate(req: GenerateReq):
    model = MODEL
    tok = TOK
    if model is None:
        return {"error": "model not loaded; place model_bundle.pt next to the app"}
    if tok is None:
        ids = torch.tensor([[c for c in req.prompt.encode("utf-8")]], dtype=torch.long)
        out = sample(
            model,
            ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=(req.top_k or None),
            top_p=(req.top_p or None),
        )
        text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    else:
        ids = torch.tensor([tok.encode(req.prompt)], dtype=torch.long)
        out = sample(
            model,
            ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=(req.top_k or None),
            top_p=(req.top_p or None),
        )
        text = tok.decode(out[0].tolist())
    return {"text": text}

