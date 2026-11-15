#!/usr/bin/env python3
"""
End-to-end dataset preparation orchestrator for STAN AI:
1. Download license-verified public-domain English sources
2. Clean, normalize, and filter documents
3. Deduplicate and split into leak-free train/val splits
4. Train custom SentencePiece BPE tokenizer (8192 vocab) on train split ONLY
5. Pack tokenized train.bin and val.bin arrays
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_binary_dataset
from data.deduplicate import deduplicate_and_split
from data.download import VERIFIED_PUBLIC_DOMAIN_BOOKS, download_corpus_sources
from data.preprocess import preprocess_corpus
from tokenizer.train_tokenizer import train_bpe_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STAN AI English corpus and BPE tokenizer.")
    parser.add_argument("--sources-limit", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()

    print("[1/5] Downloading license-verified public-domain sources...")
    manifest_path = download_corpus_sources(
        VERIFIED_PUBLIC_DOMAIN_BOOKS[: max(1, args.sources_limit)],
        output_dir=args.raw_dir,
    )

    print("[2/5] Preprocessing and filtering documents...")
    cleaned_jsonl = Path(args.processed_dir) / "cleaned.jsonl"
    prep_stats = preprocess_corpus(manifest_path, cleaned_jsonl)
    print(f"  -> {prep_stats}")

    print("[3/5] Deduplicating and splitting train/validation...")
    split_stats = deduplicate_and_split(
        cleaned_jsonl,
        output_dir=args.processed_dir,
        val_ratio=args.val_ratio,
    )
    print(f"  -> {split_stats}")

    print("[4/5] Training SentencePiece BPE tokenizer on train split ONLY...")
    tok_model, tok_cfg = train_bpe_tokenizer(
        train_files=[split_stats["train_tokenizer_corpus"]],
        validation_files=[split_stats["val_jsonl"]],
        output_dir=args.checkpoint_dir,
        vocab_size=args.vocab_size,
    )
    print(f"  -> {tok_model} ({tok_cfg})")

    print("[5/5] Packing train.bin and val.bin...")
    ds_meta = build_binary_dataset(
        train_jsonl=split_stats["train_jsonl"],
        val_jsonl=split_stats["val_jsonl"],
        tokenizer_model=tok_model,
        output_dir=args.processed_dir,
    )
    print(f"  -> Complete! Total tokens: {ds_meta['total_tokens']:,}")


if __name__ == "__main__":
    main()
