from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class STANConfig:
    """Configuration for the STAN AI decoder-only Transformer."""

    model_name: str = "STAN-AI"
    version: str = "0.1.0"
    architecture: str = "decoder_only_transformer"
    n_layers: int = 15
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    d_ff: int = 2048
    max_seq_len: int = 512
    vocab_size: int = 8192
    activation: Literal["gelu", "swiglu"] = "gelu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    bias: bool = False
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("n_layers, d_model, and n_heads must be positive integers.")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads * d_head ({self.n_heads} * {self.d_head}) must equal d_model ({self.d_model})."
            )
        if self.activation not in ("gelu", "swiglu"):
            raise ValueError(f"Unsupported activation '{self.activation}'. Use 'gelu' or 'swiglu'.")
        if self.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16.")
        if self.max_seq_len < 4:
            raise ValueError("max_seq_len must be at least 4.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "STANConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "STANConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
