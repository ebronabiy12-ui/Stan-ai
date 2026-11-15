from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer import STANTokenizer, train_bpe_tokenizer


SAMPLE_TRAIN_TEXT = "\n".join(
    [
        "The independent decoder-only Transformer predicts the next token from causal context.",
        "Public-domain literature provides diverse English syntax, vocabulary, and narrative rhythm.",
        "Tokenization splits text into subword units while preserving exact round-trip decoding.",
        "Gradient clipping and cosine learning-rate decay stabilize optimization across epochs.",
        "Evaluation measures held-out validation cross-entropy loss without leaking training samples.",
        "Checkpoints store model weights, optimizer state, scheduler state, and global step count.",
    ]
    * 40
)


def test_bpe_tokenizer_train_roundtrip_and_leakage_guard(tmp_path: Path) -> None:
    train_corpus = tmp_path / "train.txt"
    val_corpus = tmp_path / "val.txt"
    train_corpus.write_text(SAMPLE_TRAIN_TEXT, encoding="utf-8")
    val_corpus.write_text("Validation sentence kept strictly separate from training.\n", encoding="utf-8")

    model_path, config_path = train_bpe_tokenizer(
        train_files=[train_corpus],
        validation_files=[val_corpus],
        output_dir=tmp_path / "tok",
        vocab_size=512,
    )

    assert model_path.exists()
    assert config_path.exists()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["trained_on_train_split_only"] is True

    tok = STANTokenizer(model_path, config_path)
    assert tok.vocab_size == 512
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3

    sample = "STAN AI predicts tokens cleanly: 100% deterministic!"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert decoded == sample

    # Leakage guard must reject overlapping validation file
    with pytest.raises(ValueError, match="Data leakage detected"):
        train_bpe_tokenizer(
            train_files=[train_corpus],
            validation_files=[train_corpus],
            output_dir=tmp_path / "tok_leak",
            vocab_size=512,
        )
