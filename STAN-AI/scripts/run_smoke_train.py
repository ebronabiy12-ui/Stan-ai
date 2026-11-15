#!/usr/bin/env python3
"""
Synthetic Numerical Smoke-Test Training Runner for STAN AI.
WARNING: Trains on a small synthetic numerical pattern ONLY to verify
training mechanics, loss reduction, and checkpoint resume — NOT a real English model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import generate_smoke_test_dataset
from model.config import STANConfig
from training.train import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STAN AI synthetic numerical smoke training test.")
    parser.add_argument("--work-dir", default="/tmp/stan_smoke_test")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    work = Path(args.work_dir)
    smoke_meta = generate_smoke_test_dataset(work / "data", vocab_size=128, train_tokens=2048, val_tokens=512)

    model_cfg = STANConfig(
        model_name="STAN-AI-SmokeTest",
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        vocab_size=128,
    )
    train_cfg = TrainingConfig(
        seed=42,
        batch_size=4,
        grad_accum_steps=1,
        seq_len=32,
        max_steps=args.steps,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=5,
        eval_interval=10,
        eval_steps=4,
        save_interval=20,
        log_interval=5,
        train_data_path=smoke_meta["train"]["bin_path"],
        val_data_path=smoke_meta["val"]["bin_path"],
        tokenizer_path=str(work / "none.model"),
        checkpoint_dir=str(work / "checkpoints"),
        training_status_on_complete="smoke_test",
    )
    result = run_training(model_cfg, train_cfg, device_override="cpu")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
