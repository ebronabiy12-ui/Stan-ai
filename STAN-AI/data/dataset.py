from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import STANTokenizer


class PackedTokenDataset(Dataset):
    """Fixed-length autoregressive token sequence dataset backed by numpy binary arrays."""

    def __init__(self, bin_path: str | Path, seq_len: int = 512) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token binary file not found: {self.bin_path}")
        self.seq_len = seq_len
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(
                f"Token file {self.bin_path} has only {len(self.tokens)} tokens; "
                f"need > seq_len + 1 ({seq_len + 1})."
            )
        self.num_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def _encode_jsonl_to_bin(
    jsonl_path: Path,
    bin_path: Path,
    tokenizer: STANTokenizer,
) -> dict[str, Any]:
    token_ids: list[int] = []
    doc_count = 0
    licenses: set[str] = set()
    sources: set[str] = set()

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = tokenizer.encode(rec["text"], add_bos=True, add_eos=True)
            token_ids.extend(ids)
            doc_count += 1
            if "license" in rec:
                licenses.add(rec["license"])
            if "source_id" in rec:
                sources.add(rec["source_id"])

    arr = np.asarray(token_ids, dtype=np.uint16)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(bin_path)
    return {
        "documents": doc_count,
        "tokens": int(arr.size),
        "licenses": sorted(licenses),
        "sources": sorted(sources),
        "bin_path": str(bin_path),
    }


def build_binary_dataset(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    tokenizer_model: str | Path,
    output_dir: str | Path = "STAN-AI/data/processed",
) -> dict[str, Any]:
    """Tokenize split JSONL files into reusable train.bin and val.bin arrays."""
    tokenizer = STANTokenizer(tokenizer_model)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_stats = _encode_jsonl_to_bin(Path(train_jsonl), out_dir / "train.bin", tokenizer)
    val_stats = _encode_jsonl_to_bin(Path(val_jsonl), out_dir / "val.bin", tokenizer)

    metadata = {
        "is_smoke_test": False,
        "corpus_type": "english_open_license_corpus",
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": train_stats["tokens"] + val_stats["tokens"],
        "train": train_stats,
        "val": val_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def generate_smoke_test_dataset(
    output_dir: str | Path,
    vocab_size: int = 128,
    train_tokens: int = 4096,
    val_tokens: int = 512,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Create a small synthetic numerical smoke-test dataset strictly for testing
    the training loop, loss reduction, and checkpoint resume mechanics.
    NOT a real English language corpus.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Repeating predictable pattern + controlled noise so a tiny model demonstrably learns
    base_pattern = np.arange(4, min(vocab_size, 36), dtype=np.uint16)
    train_arr = np.resize(base_pattern, train_tokens).copy()
    val_arr = np.resize(base_pattern, val_tokens).copy()

    # Inject small deterministic variation outside special token IDs [0..3]
    noise_mask_train = rng.random(train_tokens) < 0.05
    train_arr[noise_mask_train] = rng.integers(4, vocab_size, size=int(noise_mask_train.sum()), dtype=np.uint16)

    train_bin = out_dir / "smoke_train.bin"
    val_bin = out_dir / "smoke_val.bin"
    train_arr.tofile(train_bin)
    val_arr.tofile(val_bin)

    metadata = {
        "is_smoke_test": True,
        "warning": "SYNTHETIC NUMERICAL SMOKE-TEST DATASET ONLY — NOT REAL LANGUAGE CORPUS.",
        "vocab_size": vocab_size,
        "total_tokens": int(train_tokens + val_tokens),
        "train": {"tokens": int(train_tokens), "bin_path": str(train_bin)},
        "val": {"tokens": int(val_tokens), "bin_path": str(val_bin)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "smoke_dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenized binary datasets for STAN AI.")
    parser.add_argument("--train-jsonl", default="STAN-AI/data/processed/train.jsonl")
    parser.add_argument("--val-jsonl", default="STAN-AI/data/processed/val.jsonl")
    parser.add_argument("--tokenizer", default="STAN-AI/checkpoints/tokenizer.model")
    parser.add_argument("--output-dir", default="STAN-AI/data/processed")
    args = parser.parse_args()

    meta = build_binary_dataset(args.train_jsonl, args.val_jsonl, args.tokenizer, args.output_dir)
    print(f"[STAN AI] Built binary dataset: total_tokens={meta['total_tokens']:,}")


if __name__ == "__main__":
    main()
