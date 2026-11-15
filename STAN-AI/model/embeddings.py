from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionalEmbeddings(nn.Module):
    """Learned token embeddings summed with learned absolute positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}."
            )
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions).unsqueeze(0)
        return self.dropout(tok_emb + pos_emb)
