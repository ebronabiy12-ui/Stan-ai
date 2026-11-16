from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def _word_shingles(text: str, k: int = 5) -> set[str]:
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def exact_deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        digest = hashlib.sha256(_normalize_for_hash(rec["text"]).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(rec)
    return unique


def near_deduplicate(
    records: list[dict[str, Any]],
    shingle_k: int = 5,
    jaccard_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    kept_shingles: list[set[str]] = []
    shingle_index: dict[str, list[int]] = {}

    for rec in records:
        shingles = _word_shingles(rec["text"], k=shingle_k)
        if not shingles:
            continue

        overlap_counts: dict[int, int] = {}
        for sh in shingles:
            for doc_idx in shingle_index.get(sh, ()):
                overlap_counts[doc_idx] = overlap_counts.get(doc_idx, 0) + 1

        is_dup = False
        sh_len = len(shingles)
        for doc_idx, inter in overlap_counts.items():
            other_len = len(kept_shingles[doc_idx])
            union = sh_len + other_len - inter
            if union > 0 and (inter / union) >= jaccard_threshold:
                is_dup = True
                break

        if not is_dup:
            new_idx = len(kept)
            kept.append(rec)
            kept_shingles.append(shingles)
            for sh in shingles:
                shingle_index.setdefault(sh, []).append(new_idx)
    return kept


def deduplicate_and_split(
    input_jsonl: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    jaccard_threshold: float = 0.85,
) -> dict[str, Any]:
    """Deduplicate cleaned JSONL documents and create leak-free train/validation splits."""
    in_path = Path(input_jsonl)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    after_exact = exact_deduplicate(records)
    after_near = near_deduplicate(after_exact, jaccard_threshold=jaccard_threshold)

    rng = random.Random(seed)
    shuffled = list(after_near)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) >= 2 else 0
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"
    train_txt = out_dir / "train_tokenizer_corpus.txt"

    with open(train_jsonl, "w", encoding="utf-8") as tf, open(train_txt, "w", encoding="utf-8") as tt:
        for rec in train_records:
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tt.write(rec["text"].replace("\n", " ") + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as vf:
        for rec in val_records:
            vf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "input_documents": len(records),
        "after_exact_dedup": len(after_exact),
        "after_near_dedup": len(after_near),
        "train_documents": len(train_records),
        "val_documents": len(val_records),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "train_tokenizer_corpus": str(train_txt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and split STAN AI corpus.")
    parser.add_argument("--input", default="STAN-AI/data/processed/cleaned.jsonl")
    parser.add_argument("--output-dir", default="STAN-AI/data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = deduplicate_and_split(args.input, args.output_dir, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[STAN AI] Deduplication & split complete: {summary}")


if __name__ == "__main__":
    main()
