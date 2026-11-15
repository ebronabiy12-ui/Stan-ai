from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (
    PackedTokenDataset,
    build_binary_dataset,
    clean_document_text,
    deduplicate_and_split,
    preprocess_corpus,
    register_local_source,
)
from tokenizer import train_bpe_tokenizer


def test_data_pipeline_clean_dedup_split_and_pack(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    doc_file = raw_dir / "sample_book.txt"
    doc_file.write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "<p>Chapter One: The mechanical clock ticked steadily across the quiet laboratory hall.</p>\n\n"
        "Engineers examined every gear, shaft, and bearing with meticulous precision during testing.\n\n"
        "Chapter One: The mechanical clock ticked steadily across the quiet laboratory hall.\n\n"
        "Another distinct paragraph describing atmospheric pressure measurements and calibration curves in detail.\n\n"
        "A third distinct paragraph documenting thermodynamic efficiency across continuous operating cycles.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***",
        encoding="utf-8",
    )

    cleaned = clean_document_text(doc_file.read_text(encoding="utf-8"), strip_gutenberg=True)
    assert "PROJECT GUTENBERG" not in cleaned
    assert "<p>" not in cleaned

    register_local_source(
        file_path=doc_file,
        source_id="sample_public_domain",
        title="Sample Public Domain Text",
        category="public_domain_book",
        license_name="Public Domain",
        attribution="Test Fixture",
        output_dir=raw_dir,
    )

    cleaned_jsonl = tmp_path / "processed" / "cleaned.jsonl"
    prep_stats = preprocess_corpus(raw_dir / "manifest.json", cleaned_jsonl, min_chars=50, target_doc_chars=120)
    assert prep_stats["kept_documents"] >= 3

    split_stats = deduplicate_and_split(cleaned_jsonl, tmp_path / "processed", val_ratio=0.33, seed=42)
    assert split_stats["train_documents"] >= 2
    assert split_stats["val_documents"] >= 1

    # Verify zero document overlap between train and val
    train_texts = {
        json.loads(line)["text"]
        for line in Path(split_stats["train_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    val_texts = {
        json.loads(line)["text"]
        for line in Path(split_stats["val_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert train_texts.isdisjoint(val_texts)

    # Train tokenizer on train corpus only
    # Expand corpus lines enough for 512 merges
    corpus_txt = Path(split_stats["train_tokenizer_corpus"])
    base_lines = corpus_txt.read_text(encoding="utf-8")
    corpus_txt.write_text((base_lines + "\n") * 50, encoding="utf-8")

    tok_model, _ = train_bpe_tokenizer(
        train_files=[corpus_txt],
        validation_files=[Path(split_stats["val_jsonl"])],
        output_dir=tmp_path / "tok",
        vocab_size=350,
    )

    meta = build_binary_dataset(
        train_jsonl=split_stats["train_jsonl"],
        val_jsonl=split_stats["val_jsonl"],
        tokenizer_model=tok_model,
        output_dir=tmp_path / "processed",
    )
    assert meta["is_smoke_test"] is False
    assert meta["total_tokens"] > 40

    ds = PackedTokenDataset(meta["train"]["bin_path"], seq_len=16)
    assert len(ds) >= 1
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
