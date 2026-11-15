from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import STANConfig
from .embeddings import TokenAndPositionalEmbeddings
from .normalization import RMSNorm


class FeedForward(nn.Module):
    """Configurable Transformer feed-forward network supporting GELU or SwiGLU."""

    def __init__(self, config: STANConfig) -> None:
        super().__init__()
        self.activation_name = config.activation
        self.dropout = nn.Dropout(config.dropout)

        if config.activation == "gelu":
            self.fc_in = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.fc_out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        elif config.activation == "swiglu":
            self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            h = F.gelu(self.fc_in(x), approximate="tanh")
            return self.dropout(self.fc_out(h))
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(h))


class TransformerBlock(nn.Module):
    """Pre-norm decoder-only Transformer block with residual connections."""

    def __init__(self, config: STANConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class STANTransformer(nn.Module):
    """STAN AI decoder-only Transformer language model initialized from random weights."""

    def __init__(self, config: STANConfig, log_params: bool = True) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenAndPositionalEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

        if log_params:
            counts = self.parameter_breakdown()
            print(
                f"[STAN AI] Initialized {config.model_name} v{config.version} "
                f"({config.activation.upper()}, {config.n_layers}L/{config.d_model}D/{config.n_heads}H) | "
                f"Trainable parameters: {counts['trainable_parameters']:,} "
                f"({counts['trainable_parameters'] / 1e6:.2f}M)"
            )

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        resid_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=resid_std)
            if hasattr(block.ffn, "fc_out"):
                nn.init.normal_(block.ffn.fc_out.weight, mean=0.0, std=resid_std)
            elif hasattr(block.ffn, "down_proj"):
                nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=resid_std)

    def count_parameters(self, trainable_only: bool = True) -> int:
        # Deduplicate tied weights via unique parameter object identity
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                total += p.numel()
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        tok_emb = self.embeddings.token_embedding.weight.numel()
        pos_emb = self.embeddings.position_embedding.weight.numel()
        blocks_params = sum(p.numel() for p in self.blocks.parameters())
        final_norm_params = sum(p.numel() for p in self.final_norm.parameters())
        lm_head_unique = 0 if self.config.tie_embeddings else self.lm_head.weight.numel()
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "token_embedding_parameters": tok_emb,
            "positional_embedding_parameters": pos_emb,
            "transformer_blocks_parameters": blocks_params,
            "final_norm_parameters": final_norm_params,
            "lm_head_unique_parameters": lm_head_unique,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        return logits, loss
