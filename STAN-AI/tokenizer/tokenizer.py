from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import sentencepiece as spm


class STANTokenizer:
    """Wrapper around STAN AI's custom SentencePiece BPE tokenizer."""

    def __init__(
        self,
        model_path: str | Path,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found at {self.model_path}")

        self.sp = spm.SentencePieceProcessor()
        loaded = self.sp.load(str(self.model_path))
        if not loaded:
            raise RuntimeError(f"Failed to load SentencePiece model from {self.model_path}")

        self.config_path = (
            Path(config_path)
            if config_path is not None
            else self.model_path.with_name("tokenizer_config.json")
        )
        self.config: dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self.config = json.load(fh)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        ids = [int(t) for t in token_ids]
        if skip_special_tokens:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [t for t in ids if t not in special]
        return self.sp.decode(ids)

    def encode_pieces(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        return list(self.sp.encode(text, out_type=str))

    def id_to_piece(self, token_id: int) -> str:
        return str(self.sp.id_to_piece(int(token_id)))

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "config": self.config,
        }
