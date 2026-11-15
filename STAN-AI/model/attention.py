from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with explicit lower-triangular masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError("n_heads * d_head must equal d_model.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal, float("-inf"))

        if attention_mask is not None:
            # attention_mask shape: (bsz, seq_len) with 1 for valid tokens and 0 for padding
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        probs = self.attn_dropout(probs)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.resid_dropout(self.o_proj(out))
