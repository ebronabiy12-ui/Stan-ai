from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data.dataset import PackedTokenDataset
from model.config import STANConfig
from model.transformer import STANTransformer
from training.checkpoint import load_checkpoint, resolve_latest_checkpoint, save_checkpoint
from training.evaluate import evaluate_model
from training.optimizer import create_adamw_optimizer
from training.scheduler import create_cosine_warmup_scheduler


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    grad_accum_steps: int = 4
    seq_len: int = 512
    max_steps: int = 50000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: bool = True
    eval_interval: int = 500
    eval_steps: int = 50
    save_interval: int = 1000
    log_interval: int = 50
    train_data_path: str = "STAN-AI/data/processed/train.bin"
    val_data_path: str = "STAN-AI/data/processed/val.bin"
    tokenizer_path: str = "STAN-AI/checkpoints/tokenizer.model"
    checkpoint_dir: str = "STAN-AI/checkpoints"
    training_status_on_complete: str = "trained"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in valid})


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    model_config: STANConfig,
    train_config: TrainingConfig,
    resume_from: Optional[str | Path] = None,
    device_override: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute STAN AI autoregressive next-token training loop with:
    - AdamW + cosine warmup/decay
    - Gradient accumulation + gradient clipping
    - Optional CUDA mixed precision
    - Periodic validation evaluation
    - Resumable checkpoint persistence
    """
    set_reproducible_seed(train_config.seed)

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PackedTokenDataset(train_config.train_data_path, seq_len=train_config.seq_len)
    val_dataset = PackedTokenDataset(train_config.val_data_path, seq_len=train_config.seq_len)
    dataset_tokens = len(train_dataset.tokens) + len(val_dataset.tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = STANTransformer(model_config, log_params=True).to(device)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
    )
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        max_lr=train_config.learning_rate,
        min_lr=train_config.min_learning_rate,
    )

    use_amp = bool(train_config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    start_step = 0
    history: list[dict[str, Any]] = []

    if resume_from is not None:
        ckpt_path = (
            resolve_latest_checkpoint(train_config.checkpoint_dir)
            if str(resume_from) == "auto"
            else Path(resume_from)
        )
        if ckpt_path is not None and ckpt_path.exists():
            payload = load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=device,
            )
            start_step = int(payload.get("global_step", 0))
            history = list(payload.get("training_history", []))
            print(f"[STAN AI] Resumed training from {ckpt_path} at global_step={start_step}")

    model.train()
    train_iter = iter(train_loader)
    global_step = start_step
    last_train_loss: Optional[float] = None
    last_val_loss: Optional[float] = None

    while global_step < train_config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(train_config.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(x, targets=y)
                assert loss is not None
                scaled_loss = loss / float(train_config.grad_accum_steps)

            if scaler is not None and use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += float(loss.item()) / float(train_config.grad_accum_steps)

        if scaler is not None and use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

        lr = scheduler.step()
        global_step += 1
        last_train_loss = accum_loss

        should_eval = (
            global_step % train_config.eval_interval == 0
            or global_step == train_config.max_steps
        )
        val_loss_val: Optional[float] = None
        if should_eval:
            eval_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                max_batches=train_config.eval_steps,
                use_amp=use_amp,
            )
            val_loss_val = eval_metrics["val_loss"]
            if val_loss_val is not None:
                last_val_loss = val_loss_val

        if global_step % train_config.log_interval == 0 or should_eval or global_step == 1:
            entry = {
                "step": global_step,
                "train_loss": round(accum_loss, 6),
                "val_loss": round(val_loss_val, 6) if val_loss_val is not None else None,
                "learning_rate": lr,
            }
            history.append(entry)

        if global_step % train_config.save_interval == 0 or global_step == train_config.max_steps:
            status = (
                train_config.training_status_on_complete
                if global_step >= train_config.max_steps
                else "training"
            )
            save_checkpoint(
                checkpoint_dir=train_config.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                model_config=model_config,
                training_config=train_config.to_dict(),
                training_history=history,
                training_status=status,
                tokenizer_path=train_config.tokenizer_path,
                dataset_tokens=dataset_tokens,
            )

    return {
        "global_step": global_step,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "dataset_tokens": dataset_tokens,
        "checkpoint_dir": train_config.checkpoint_dir,
        "history": history,
        "status": train_config.training_status_on_complete if global_step >= train_config.max_steps else "training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STAN AI Transformer from scratch.")
    parser.add_argument("--model-config", default="STAN-AI/configs/model.yaml")
    parser.add_argument("--train-config", default="STAN-AI/configs/training.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt or 'auto'.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model_cfg = STANConfig.from_yaml(args.model_config)
    train_cfg = TrainingConfig.from_yaml(args.train_config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps

    summary = run_training(
        model_config=model_cfg,
        train_config=train_cfg,
        resume_from=args.resume,
        device_override=args.device,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
