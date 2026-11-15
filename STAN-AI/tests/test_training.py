from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import generate_smoke_test_dataset
from model import STANConfig, STANTransformer
from training import (
    TrainingConfig,
    create_adamw_optimizer,
    create_cosine_warmup_scheduler,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


def test_scheduler_warmup_and_cosine_decay() -> None:
    cfg = STANConfig(n_layers=1, d_model=32, n_heads=2, d_head=16, d_ff=64, max_seq_len=16, vocab_size=64)
    model = STANTransformer(cfg, log_params=False)
    opt = create_adamw_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    sched = create_cosine_warmup_scheduler(opt, warmup_steps=4, max_steps=20, max_lr=1e-3, min_lr=1e-4)

    lrs = [sched.current_lr()]
    for _ in range(20):
        lrs.append(sched.step())

    assert lrs[0] < lrs[3]
    assert pytest.approx(lrs[3], rel=1e-4) == 1e-3
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-4)


def test_checkpoint_save_load_and_resume_smoke_training(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_data"
    smoke_meta = generate_smoke_test_dataset(smoke_dir, vocab_size=128, train_tokens=1024, val_tokens=256)
    assert smoke_meta["is_smoke_test"] is True

    model_cfg = STANConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        vocab_size=128,
    )
    ckpt_dir = tmp_path / "checkpoints"

    # Phase 1: Train 15 steps on synthetic numerical smoke-test data
    train_cfg_p1 = TrainingConfig(
        seed=42,
        batch_size=4,
        grad_accum_steps=1,
        seq_len=32,
        max_steps=15,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=3,
        eval_interval=5,
        eval_steps=4,
        save_interval=15,
        log_interval=1,
        train_data_path=smoke_meta["train"]["bin_path"],
        val_data_path=smoke_meta["val"]["bin_path"],
        tokenizer_path=str(tmp_path / "nonexistent_tokenizer.model"),
        checkpoint_dir=str(ckpt_dir),
        training_status_on_complete="smoke_test",
    )
    res1 = run_training(model_cfg, train_cfg_p1, device_override="cpu")
    assert res1["global_step"] == 15
    assert (ckpt_dir / "latest.pt").exists()
    first_loss = res1["history"][0]["train_loss"]
    step15_loss = res1["history"][-1]["train_loss"]
    assert step15_loss < first_loss

    # Verify standalone checkpoint loading restores exact weights and step
    fresh_model = STANTransformer(model_cfg, log_params=False)
    payload = load_checkpoint(ckpt_dir / "latest.pt", fresh_model)
    assert payload["global_step"] == 15
    assert payload["training_status"] == "smoke_test"

    # Phase 2: Resume from step 15 to step 30
    train_cfg_p2 = TrainingConfig(
        seed=42,
        batch_size=4,
        grad_accum_steps=1,
        seq_len=32,
        max_steps=30,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=3,
        eval_interval=5,
        eval_steps=4,
        save_interval=15,
        log_interval=1,
        train_data_path=smoke_meta["train"]["bin_path"],
        val_data_path=smoke_meta["val"]["bin_path"],
        tokenizer_path=str(tmp_path / "nonexistent_tokenizer.model"),
        checkpoint_dir=str(ckpt_dir),
        training_status_on_complete="smoke_test",
    )
    res2 = run_training(model_cfg, train_cfg_p2, resume_from="auto", device_override="cpu")
    assert res2["global_step"] == 30
    assert res2["final_train_loss"] is not None
    assert res2["final_train_loss"] < step15_loss
    assert (ckpt_dir / "training_history.json").exists()
