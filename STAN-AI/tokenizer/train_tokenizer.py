from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import sentencepiece as spm


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def train_bpe_tokenizer(
    train_files: Iterable[str | Path],
    output_dir: str | Path,
    model_prefix: str = "tokenizer",
    vocab_size: int = 8192,
    character_coverage: float = 0.9995,
    byte_fallback: bool = True,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
    validation_files: Optional[Iterable[str | Path]] = None,
    max_sentence_length: int = 8192,
) -> tuple[Path, Path]:
    """
    Train a custom SentencePiece BPE tokenizer strictly from STAN AI's training corpus.
    Enforces train/validation separation so validation data never leaks into vocabulary training.
    """
    train_paths = [Path(p).resolve() for p in train_files]
    if not train_paths:
        raise ValueError("At least one training corpus file is required to train the tokenizer.")
    for p in train_paths:
        if not p.exists():
            raise FileNotFoundError(f"Training corpus file not found: {p}")

    val_paths = [Path(p).resolve() for p in (validation_files or [])]
    train_hashes = {_sha256_file(p) for p in train_paths}
    for vp in val_paths:
        if vp in train_paths:
            raise ValueError(f"Data leakage detected: validation file {vp} is also in train_files.")
        if vp.exists() and _sha256_file(vp) in train_hashes:
            raise ValueError(f"Data leakage detected: validation file {vp} matches a training file checksum.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / model_prefix

    input_arg = ",".join(str(p) for p in train_paths)
    spm.SentencePieceTrainer.train(
        input=input_arg,
        model_prefix=str(prefix_path),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        split_digits=True,
        allow_whitespace_only_pieces=True,
        max_sentence_length=max_sentence_length,
        normalization_rule_name="nfkc",
    )

    model_path = Path(f"{prefix_path}.model")
    config_path = out_dir / "tokenizer_config.json"

    line_count = 0
    for p in train_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in fh:
                line_count += 1

    config_payload = {
        "model_name": "STAN-AI-BPE",
        "tokenizer_type": "sentencepiece_bpe",
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "special_tokens": {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
        },
        "byte_fallback": byte_fallback,
        "character_coverage": character_coverage,
        "trained_on_train_split_only": True,
        "train_files": [str(p.name) for p in train_paths],
        "train_file_sha256": sorted(train_hashes),
        "num_training_lines": line_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2)

    return model_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STAN AI SentencePiece BPE tokenizer from scratch.")
    parser.add_argument("--input", nargs="+", required=True, help="Training text files (train split only).")
    parser.add_argument("--val-check", nargs="*", default=[], help="Validation files to verify zero leakage.")
    parser.add_argument("--output-dir", default="STAN-AI/checkpoints", help="Output directory for tokenizer files.")
    parser.add_argument("--prefix", default="tokenizer", help="Output file prefix.")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target BPE vocabulary size.")
    args = parser.parse_args()

    model_path, config_path = train_bpe_tokenizer(
        train_files=args.input,
        output_dir=args.output_dir,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        validation_files=args.val_check,
    )
    print(f"[STAN AI] Trained BPE tokenizer -> {model_path}")
    print(f"[STAN AI] Saved tokenizer config -> {config_path}")


if __name__ == "__main__":
    main()
