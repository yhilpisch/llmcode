from __future__ import annotations

"""
GPT assembly: token/position embeddings, a stack of Transformer blocks,
and a language‑model head. Kept small and readable to align with the book’s
step‑by‑step narrative.

Key choices
-----------
- Pre‑norm blocks (LayerNorm before sublayers) as in Chapter 8.
- Learned positional embeddings by default; optional sinusoidal positions.
- Optional weight tying between token embeddings and LM head.
- Causal mask is always applied; padding mask is optionally combined when
  a `pad_id` is provided (or an explicit `attention_mask`).

Shapes
------
Inputs are token ids of shape [B, T]; hidden/state tensors stay [B, T, D].
Logits are [B, T, V].
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Support running scripts from repo root by importing neighbor modules directly.
try:
    from ch08_transformer import (  # type: ignore
        TransformerBlock,
        sinusoidal_positions,
    )
except Exception:
    from code.ch08_transformer import (  # type: ignore
        TransformerBlock,
        sinusoidal_positions,
    )


@dataclass
class GPTConfig:
    """Hyperparameters for a small, readable GPT.

    - vocab_size: number of tokens in the vocabulary
    - block_size: maximum sequence length (context window)
    - d_model: model (embedding) dimension
    - n_head: number of attention heads per block
    - n_layer: number of transformer blocks
    - d_ff: feed‑forward hidden dimension (often 4 * d_model)
    - dropout: dropout rate in MHA/FFN
    - pos_type: 'learned' (GPT‑style) or 'sinusoidal' (chapter 8 option)
    - tie_weights: whether to tie LM head weight with token embeddings
    """

    vocab_size: int
    block_size: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    pos_type: str = "learned"  # or "sinusoidal"
    tie_weights: bool = True


class GPT(nn.Module):
    """A compact GPT‑style language model composed from Chapter 8 blocks.

    Forward signature:
        logits, loss = model(input_ids, targets=None, attention_mask=None, pad_id=None)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        V, Tm, D = cfg.vocab_size, cfg.block_size, cfg.d_model

        # Embeddings: tokens and positions
        self.tok_emb = nn.Embedding(V, D)
        if cfg.pos_type == "learned":
            self.pos_emb = nn.Embedding(Tm, D)
        else:
            self.pos_emb = None  # we'll add sinusoidal positions on the fly

        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    D,
                    cfg.n_head,
                    cfg.d_ff,
                    cfg.dropout,
                )
                for _ in range(cfg.n_layer)
            ]
        )

        # Final normalization and LM head
        self.norm_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)

        # Optional weight tying: share weights between embedding and LM head
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Small‑norm initialization consistent with readable training.

        GPT‑2 uses ~N(0, 0.02) for embeddings and projection weights. We follow
        a similar pattern here for stability at small scales.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _build_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pad_id: Optional[int],
    ) -> torch.Tensor:
        """Combine causal and optional padding masks into [B, T, T].

        - causal: lower‑triangular ones
        - padding: 1 for tokens, 0 for pads (derived from input or provided)
        """
        B, T = input_ids.size(0), input_ids.size(1)
        device = input_ids.device
        causal = torch.tril(torch.ones(T, T, device=device))  # [T, T]

        pad_mask_bt: Optional[torch.Tensor] = None
        if attention_mask is not None:
            pad_mask_bt = attention_mask.float()  # [B, T]
        elif pad_id is not None:
            # Derive from token ids: 1 for tokens, 0 for PAD
            pad_mask_bt = (input_ids != pad_id).float()  # [B, T]

        if pad_mask_bt is None:
            # No padding info: return causal broadcasted to [B, T, T]
            return causal.unsqueeze(0).expand(B, -1, -1)
        else:
            # Broadcast multiply: [B, 1, T] * [T, T] -> [B, T, T]
            return pad_mask_bt[:, None, :] * causal

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pad_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute logits (and optional loss) for a batch of token ids.

        Args:
            input_ids: LongTensor [B, T], token ids (T <= block_size)
            targets: optional LongTensor [B, T] for next‑token loss
            attention_mask: optional Float/Bool [B, T], 1 for tokens, 0 for pads
            pad_id: optional int, used only to set ignore_index in loss

        Returns:
            logits: [B, T, V]
            loss:   scalar tensor or None
        """
        B, T = input_ids.size()
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}. "
            f"Slice prompts to the last {self.cfg.block_size} tokens."
        )

        device = input_ids.device

        # Token + positional embeddings
        x = self.tok_emb(input_ids)  # [B, T, D]

        if self.cfg.pos_type == "learned":
            positions = torch.arange(T, device=device)[None, :]  # [1, T]
            x = x + self.pos_emb(positions)  # [B, T, D]
        else:
            pe = sinusoidal_positions(
                T,
                self.cfg.d_model,
                device=device,
            )  # [T, D]
            x = x + pe[None, :, :]  # [B, T, D]

        x = self.drop(x)

        # Build causal (and optional padding) mask once
        mask_btt = self._build_mask(
            input_ids,
            attention_mask,
            pad_id,
        )  # [B, T, T]

        # Pass through stacked Transformer blocks
        for block in self.blocks:
            x = block(x, mask_btt)

        # Final norm and LM head projection
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B, T, V]

        loss = None
        if targets is not None:
            # Flatten [B, T, V] -> [B*T, V] and [B, T] -> [B*T]
            logits_flat = logits.reshape(B * T, -1)
            targets_flat = targets.reshape(B * T)
            ignore = pad_id if pad_id is not None else -100
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore)

        return logits, loss


__all__ = ["GPTConfig", "GPT"]
