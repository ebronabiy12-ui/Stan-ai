#!/usr/bin/env python3
"""
Prepare a verified open-license English Stage 2 Instruction Tuning dataset (sft_instructions.jsonl).
Uses OpenAssistant OASST1 (Apache-2.0) when 'datasets' is installed (Colab/Kaggle),
verifying license metadata before emitting conversation records.
Never fabricates synthetic LLM training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download import verify_license
from data.sft_dataset import DEFAULT_SYSTEM_ANCHOR


def build_oasst1_sft_jsonl(
    output_jsonl: str | Path,
    max_pairs: int = 8000,
) -> dict[str, int | str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace 'datasets' package is required to fetch OpenAssistant/oasst1. "
            "Install with: pip install datasets"
        ) from exc

    license_name = "Apache-2.0"
    verify_license(license_name)

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id: dict[str, dict] = {}
    for row in ds:
        if row.get("lang") == "en" and not row.get("deleted", False):
            by_id[row["message_id"]] = row

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for msg_id, row in by_id.items():
            if row.get("role") != "assistant":
                continue
            parent_id = row.get("parent_id")
            if not parent_id or parent_id not in by_id:
                continue
            parent = by_id[parent_id]
            if parent.get("role") != "prompter":
                continue
            user_text = str(parent.get("text", "")).strip()
            asst_text = str(row.get("text", "")).strip()
            if len(user_text) < 8 or len(asst_text) < 16:
                continue

            record = {
                "source_id": f"oasst1_{msg_id}",
                "license": license_name,
                "system": DEFAULT_SYSTEM_ANCHOR,
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written >= max_pairs:
                break

    return {"written_examples": written, "license": license_name, "output_path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare license-verified Stage 2 SFT instruction corpus.")
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "sft_instructions.jsonl"))
    parser.add_argument("--max-pairs", type=int, default=8000)
    args = parser.parse_args()

    stats = build_oasst1_sft_jsonl(args.output, max_pairs=args.max_pairs)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
