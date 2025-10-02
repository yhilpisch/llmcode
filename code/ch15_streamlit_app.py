from __future__ import annotations

"""Streamlit app for sampling from an exported GPT bundle (Chapter 15)."""

import sys
from pathlib import Path
import streamlit as st
import torch

# Allow importing modules from code/
sys.path.append(str(Path(__file__).resolve().parent))
from ch09_gpt import GPT, GPTConfig  # type: ignore
from ch11_sampling import sample  # type: ignore
from ch6_tokenize import SimpleTokenizer, Vocab  # type: ignore


@st.cache_resource
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


st.title("Mini‑GPT Sampler")
bundle_path = st.text_input("Bundle path", "model_bundle.pt")
prompt = st.text_area("Prompt", "Hello")
col1, col2, col3 = st.columns(3)
with col1:
    max_new = st.number_input("Max new tokens", 1, 512, 80)
with col2:
    temp = st.slider("Temperature", 0.0, 1.5, 0.9, 0.05)
with col3:
    top_p = st.slider("Top‑p", 0.0, 1.0, 0.95, 0.05)
top_k = st.slider("Top‑k (0=off)", 0, 200, 0, 5)

if st.button("Generate"):
    try:
        model, tok = load_bundle(bundle_path)
    except Exception as e:
        st.error(f"Failed to load bundle: {e}")
    else:
        if tok is None:
            ids = torch.tensor([[c for c in prompt.encode("utf-8")]], dtype=torch.long)
            out = sample(
                model,
                ids,
                max_new_tokens=int(max_new),
                temperature=float(temp),
                top_k=(int(top_k) or None),
                top_p=(float(top_p) or None),
            )
            text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
        else:
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
            out = sample(
                model,
                ids,
                max_new_tokens=int(max_new),
                temperature=float(temp),
                top_k=(int(top_k) or None),
                top_p=(float(top_p) or None),
            )
            text = tok.decode(out[0].tolist())
        st.subheader("Output")
        st.write(text)

