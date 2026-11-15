from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from model.config import STANConfig


def resolve_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    ckpt_dir = Path(checkpoint_dir)
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    if candidates:
        return candidates[-1]
    return None


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    global_step: int,
    model_config: STANConfig,
    training_config: dict[str, Any],
    training_history: list[dict[str, Any]],
    training_status: str = "training",
    tokenizer_path: Optional[str | Path] = None,
    dataset_tokens: int = 0,
) -> Path:
    """
    Atomically save STAN AI checkpoint with weights, optimizer, scheduler, config,
    tokenizer assets, and training metrics history.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_version = f"step-{global_step:07d}"
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "model_config": model_config.to_dict(),
        "training_config": training_config,
        "training_status": training_status,
        "checkpoint_version": checkpoint_version,
        "dataset_tokens": int(dataset_tokens),
        "training_history": training_history,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    step_path = ckpt_dir / f"step_{global_step:07d}.pt"
    tmp_path = ckpt_dir / f".tmp_step_{global_step:07d}.pt"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, step_path)

    latest_path = ckpt_dir / "latest.pt"
    tmp_latest = ckpt_dir / ".tmp_latest.pt"
    torch.save(payload, tmp_latest)
    os.replace(tmp_latest, latest_path)

    # Persist readable config, history, and manifest JSON files alongside .pt weights
    model_config.to_yaml(ckpt_dir / "model_config.yaml")

    if tokenizer_path is not None:
        tok_src = Path(tokenizer_path)
        if tok_src.exists():
            tok_dst = ckpt_dir / "tokenizer.model"
            if tok_src.resolve() != tok_dst.resolve():
                shutil.copy2(tok_src, tok_dst)
            tok_cfg_src = tok_src.with_name("tokenizer_config.json")
            if tok_cfg_src.exists():
                tok_cfg_dst = ckpt_dir / "tokenizer_config.json"
                if tok_cfg_src.resolve() != tok_cfg_dst.resolve():
                    shutil.copy2(tok_cfg_src, tok_cfg_dst)

    history_payload = {
        "training_status": training_status,
        "global_step": int(global_step),
        "max_steps": int(training_config.get("max_steps", 0)),
        "dataset_tokens": int(dataset_tokens),
        "checkpoint_version": checkpoint_version,
        "saved_at": payload["saved_at"],
        "history": training_history,
    }
    with open(ckpt_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history_payload, fh, indent=2)

    return step_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load STAN AI checkpoint and restore model, optimizer, scheduler, and scaler states."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    payload = torch.load(ckpt_file, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    return payload
