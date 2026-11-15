#!/usr/bin/env python3
"""
Kaggle Notebook / Cloud GPU Training Entrypoint for STAN AI.
Supports automatic resume from latest checkpoint after Kaggle 9-12h session timeouts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.config import STANConfig
from training.train import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-compatible resumable training launcher for STAN AI.")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "training.yaml"))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default="auto", help="'auto' resumes from latest.pt if present.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    print(f"[STAN AI] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[STAN AI] GPU: {torch.cuda.get_device_name(0)}")

    model_cfg = STANConfig.from_yaml(args.model_config)
    train_cfg = TrainingConfig.from_yaml(args.train_config)
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size

    summary = run_training(model_cfg, train_cfg, resume_from=args.resume)
    print(f"[STAN AI] Finished at step {summary['global_step']} | status={summary['status']}")


if __name__ == "__main__":
    main()
