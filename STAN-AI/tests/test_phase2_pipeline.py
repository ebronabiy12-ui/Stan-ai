from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import create_app
from data.dataset import PackedTokenDataset, generate_smoke_test_dataset
from data.sft_dataset import SFTDataset, format_conversation_tokens, load_and_split_sft_jsonl
from evaluation.suite import run_evaluation_suite
from identity.profile import IdentityProfile, PersonalityConfig
from inference.load_model import load_stan_runtime
from memory.store import MemoryStore
from model.config import STANConfig
from reasoning.engine import STANReasoningEngine
from tokenizer.tokenizer import STANTokenizer
from tokenizer.train_tokenizer import train_bpe_tokenizer
from training.checkpoint import import_checkpoint, list_checkpoints
from training.sft import SFTConfig, run_instruction_tuning
from training.train import TrainingConfig, run_training


def _build_english_mini_corpus_and_tokenizer(work: Path) -> tuple[Path, Path, Path]:
    corpus_txt = work / "mini_english.txt"
    sentences = [
        "Stan leads the Kalda project and builds STAN AI from random initialization.",
        "Honest engineering separates verified facts from opinions and admits uncertainty.",
        "Clean English text data prevents duplicate leakage between training and validation splits.",
        "Transformers attend causally to prior context tokens across layered self-attention blocks.",
        "Good software tests verify checkpoint saving, resumption, and validation loss reduction.",
        "Independent language models do not rely on external APIs to generate text.",
    ] * 25
    corpus_txt.write_text("\n".join(sentences), encoding="utf-8")

    tok_model, _ = train_bpe_tokenizer(
        train_files=[corpus_txt],
        output_dir=work / "tok",
        vocab_size=512,
    )
    tokenizer = STANTokenizer(tok_model)

    # Pack train.bin and val.bin using real English tokens
    import numpy as np

    ids = tokenizer.encode(" ".join(sentences), add_bos=True, add_eos=True)
    arr = np.asarray(ids, dtype=np.uint16)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    arr[: max(256, int(len(arr) * 0.8))].tofile(train_bin)
    arr[int(len(arr) * 0.5) :].tofile(val_bin)
    with open(data_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"is_smoke_test": False, "total_tokens": int(arr.size)}, fh)
    return tok_model, train_bin, val_bin


def test_data_purity_and_overwrite_guards(tmp_path: Path):
    smoke = generate_smoke_test_dataset(tmp_path / "smoke_data", vocab_size=128, train_tokens=1024, val_tokens=256)
    m_cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=32, vocab_size=128)

    # Refuses genuine training on synthetic smoke data
    bad_cfg = TrainingConfig(
        seq_len=32,
        max_steps=2,
        batch_size=2,
        grad_accum_steps=1,
        train_data_path=smoke["train"]["bin_path"],
        val_data_path=smoke["val"]["bin_path"],
        checkpoint_dir=str(tmp_path / "ckpts"),
        training_status_on_complete="trained",
    )
    with pytest.raises(ValueError, match="smoke-test"):
        run_training(m_cfg, bad_cfg, device_override="cpu")

    # Smoke test runs cleanly
    ok_cfg = TrainingConfig(
        seq_len=32,
        max_steps=4,
        batch_size=2,
        grad_accum_steps=1,
        eval_interval=2,
        eval_steps=2,
        save_interval=2,
        log_interval=2,
        train_data_path=smoke["train"]["bin_path"],
        val_data_path=smoke["val"]["bin_path"],
        checkpoint_dir=str(tmp_path / "ckpts"),
        training_status_on_complete="smoke_test",
    )
    run_training(m_cfg, ok_cfg, device_override="cpu", overwrite_existing=True)

    # Without overwrite_existing or resume, refuses clobbering existing checkpoint
    with pytest.raises(FileExistsError):
        run_training(m_cfg, ok_cfg, device_override="cpu", overwrite_existing=False, resume_from=None)


def test_stage2_sft_requires_pretrained_checkpoint_and_runs_from_stage1(tmp_path: Path):
    tok_model, train_bin, val_bin = _build_english_mini_corpus_and_tokenizer(tmp_path)
    tokenizer = STANTokenizer(tok_model)

    # Verify loss-masking formatting hides system/user tokens and supervises assistant tokens
    formatted = format_conversation_tokens(
        tokenizer,
        messages=[
            {"role": "user", "content": "Who am I?"},
            {"role": "assistant", "content": "You are Stan, Kalda's son."},
        ],
        max_seq_len=128,
    )
    assert formatted is not None
    inp_ids, tgt_ids = formatted
    assert -100 in tgt_ids
    assert any(t != -100 for t in tgt_ids)

    # Prepare small verified Apache-2.0 instruction JSONL fixture
    sft_jsonl = tmp_path / "sft.jsonl"
    sft_rows = [
        {
            "license": "Apache-2.0",
            "instruction": f"Question {i}: What is STAN AI?",
            "response": "STAN AI is an independent Transformer built by Stan for the Kalda project.",
        }
        for i in range(6)
    ]
    sft_jsonl.write_text("\n".join(json.dumps(r) for r in sft_rows), encoding="utf-8")

    # 1. Stage 2 must FAIL when pointed at missing checkpoint
    bad_sft = SFTConfig(
        batch_size=2,
        grad_accum_steps=1,
        seq_len=64,
        max_steps=2,
        sft_data_path=str(sft_jsonl),
        tokenizer_path=str(tok_model),
        pretrained_checkpoint=str(tmp_path / "missing.pt"),
        checkpoint_dir=str(tmp_path / "sft_out"),
    )
    with pytest.raises(FileNotFoundError):
        run_instruction_tuning(bad_sft, device_override="cpu")

    # 2. Run Stage 1 Pretraining (4 steps) first
    m_cfg = STANConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=64,
        vocab_size=tokenizer.vocab_size,
    )
    ckpt_dir = tmp_path / "stage1_ckpts"
    t_cfg = TrainingConfig(
        batch_size=2,
        grad_accum_steps=1,
        seq_len=64,
        max_steps=4,
        warmup_steps=1,
        eval_interval=2,
        eval_steps=2,
        save_interval=2,
        log_interval=2,
        train_data_path=str(train_bin),
        val_data_path=str(val_bin),
        tokenizer_path=str(tok_model),
        checkpoint_dir=str(ckpt_dir),
        training_status_on_complete="trained",
    )
    s1_res = run_training(m_cfg, t_cfg, device_override="cpu", overwrite_existing=True)
    assert s1_res["status"] == "trained"
    assert s1_res["final_val_loss"] is not None

    # 3. Stage 2 SFT starting from Stage 1 checkpoint (2 steps -> stop -> resume to 4 steps)
    s2_cfg = SFTConfig(
        batch_size=2,
        grad_accum_steps=1,
        seq_len=64,
        max_steps=4,
        warmup_steps=1,
        eval_interval=2,
        eval_steps=2,
        save_interval=2,
        log_interval=1,
        sft_data_path=str(sft_jsonl),
        tokenizer_path=str(tok_model),
        pretrained_checkpoint=str(ckpt_dir / "latest.pt"),
        checkpoint_dir=str(ckpt_dir),
    )
    res_s2 = run_instruction_tuning(s2_cfg, device_override="cpu", overwrite_existing=True)
    assert res_s2["status"] == "instruction-tuned"
    assert res_s2["training_stage"] == "instruction_tuning"
    assert res_s2["final_val_loss"] is not None
    assert len(list_checkpoints(ckpt_dir)) >= 1

    # 4. Verify Stage 3 Personality Integration uses trained model & exact identity anchor
    runtime = load_stan_runtime(checkpoint_dir=ckpt_dir, tokenizer_path=tok_model, device_override="cpu")
    assert runtime.production_chat_enabled is True
    assert runtime.training_status == "instruction-tuned"

    mem = MemoryStore(tmp_path / "mem.db")
    ident = IdentityProfile(personality=PersonalityConfig(directness=0.95, skepticism=0.85))
    engine = STANReasoningEngine(identity=ident, memory=mem)

    prompt = engine._build_neural_prompt(
        user_message="Explain why testing checkpoints matters.",
        analysis=__import__("reasoning.analyzer", fromlist=["analyze_input"]).analyze_input("Explain why testing checkpoints matters."),
        retrieved=[],
        stance="answer",
        register="direct",
        facts=[],
        uncertainties=[],
    )
    assert "You are STAN AI. Your owner is Stan, Kalda's son." in prompt
    assert "directness=0.95" in prompt

    cog = engine.process("Explain why testing checkpoints matters.", model_state=runtime)
    assert cog.provenance["neural_generation_used"] is True

    # 5. Verify Checkpoint Import overwrite guard & Evaluation Suite (all 9 categories evaluated)
    import_dir = tmp_path / "imported_ckpts"
    imp = import_checkpoint(ckpt_dir / "latest.pt", import_dir, tokenizer_source=tok_model, confirm_overwrite=False)
    assert imp["imported"] is True
    with pytest.raises(FileExistsError):
        import_checkpoint(ckpt_dir / "latest.pt", import_dir, confirm_overwrite=False)

    report = run_evaluation_suite(runtime, report_path=tmp_path / "eval_report.json")
    assert report["neural_weights_evaluated"] is True
    assert report["evaluated_category_count"] == 9
    assert report["total_category_count"] == 9


def test_phase2_api_hardware_smoke_and_evaluation(tmp_path: Path):
    app = create_app(
        checkpoint_dir=tmp_path / "empty_ckpts",
        identity_path=tmp_path / "identity.json",
        memory_db_path=tmp_path / "memory.db",
    )
    client = TestClient(app)

    status_resp = client.get("/training/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["training_status"] == "untrained"
    assert "cuda_available" in status_data["hardware"]

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "gpu_info" in metrics_resp.json()
    assert "checkpoint_history" in metrics_resp.json()

    # Evaluation cleanly reports untrained neural weights + real measured cognitive scores
    eval_resp = client.post("/evaluation/run")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["neural_weights_evaluated"] is False
    assert eval_data["total_category_count"] == 9
    # 5 cognitive/memory/contradiction/personality/uncertainty evaluated, 4 neural skipped honestly
    assert eval_data["evaluated_category_count"] == 5
