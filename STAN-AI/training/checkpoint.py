from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from model.config import STANConfig
from model.transformer import STANTransformer


def resolve_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    ckpt_dir = Path(checkpoint_dir)
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    if candidates:
        return candidates[-1]
    return None


def validate_checkpoint_overwrite(
    checkpoint_dir: str | Path,
    overwrite_existing: bool = False,
    resume_from: Optional[str | Path] = None,
) -> None:
    """Prevent silently overwriting an existing checkpoint unless resuming or explicitly confirmed."""
    if resume_from is not None or overwrite_existing:
        return
    latest = resolve_latest_checkpoint(checkpoint_dir)
    if latest is not None and latest.exists():
        raise FileExistsError(
            f"Existing checkpoint found at {latest}. Pass resume='auto' to continue or set "
            "overwrite_existing=True to confirm replacing existing weights."
        )


def list_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    """Return sorted inventory of saved step checkpoints without loading full tensors to GPU."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    files = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    if not files and (ckpt_dir / "latest.pt").exists():
        files = [ckpt_dir / "latest.pt"]

    for f in files[:30]:
        info: dict[str, Any] = {
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "global_step": None,
            "training_status": None,
            "training_stage": None,
            "checkpoint_version": f.stem,
            "saved_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        try:
            payload = torch.load(f, map_location="cpu", weights_only=False)
            info["global_step"] = int(payload.get("global_step", 0))
            info["training_status"] = payload.get("training_status")
            info["training_stage"] = payload.get("training_stage")
            info["checkpoint_version"] = payload.get("checkpoint_version", f.stem)
            if payload.get("saved_at"):
                info["saved_at"] = payload["saved_at"]
        except Exception as exc:
            info["error"] = str(exc)
        records.append(info)
    return records


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
    training_stage: str = "pretraining",
    current_learning_rate: Optional[float] = None,
) -> Path:
    """
    Atomically save STAN AI checkpoint with weights, optimizer, scheduler, config,
    tokenizer assets, stage metadata, and training metrics history.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_version = f"step-{global_step:07d}"
    if current_learning_rate is None and training_history:
        current_learning_rate = training_history[-1].get("learning_rate")

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "model_config": model_config.to_dict(),
        "training_config": training_config,
        "training_status": training_status,
        "training_stage": training_stage,
        "current_learning_rate": current_learning_rate,
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
        "training_stage": training_stage,
        "current_learning_rate": current_learning_rate,
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


def import_checkpoint(
    source_checkpoint: str | Path,
    target_checkpoint_dir: str | Path,
    tokenizer_source: Optional[str | Path] = None,
    confirm_overwrite: bool = False,
) -> dict[str, Any]:
    """
    Validate and import a STAN AI checkpoint trained on Colab/Kaggle into the local
    STAN-AI/checkpoints directory. Refuses non-STAN weights or unconfirmed overwrites.
    """
    src = Path(source_checkpoint)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Source checkpoint file not found: {src}")

    ckpt_dir = Path(target_checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_latest = ckpt_dir / "latest.pt"
    if target_latest.exists() and not confirm_overwrite:
        raise FileExistsError(
            f"Destination checkpoint already exists at {target_latest}. "
            "Confirm overwrite explicitly before replacing active weights."
        )

    payload = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "model_config" not in payload:
        raise ValueError("Invalid checkpoint format: missing 'model_state_dict' or 'model_config'.")

    cfg = STANConfig.from_dict(payload["model_config"])
    # Structural compatibility verification against STANTransformer
    test_model = STANTransformer(cfg, log_params=False)
    test_model.load_state_dict(payload["model_state_dict"])
    param_count = test_model.count_parameters(trainable_only=True)
    del test_model

    global_step = int(payload.get("global_step", 0))
    step_name = f"step_{global_step:07d}.pt"
    tmp_latest = ckpt_dir / ".tmp_import_latest.pt"
    shutil.copy2(src, tmp_latest)
    os.replace(tmp_latest, target_latest)

    step_dst = ckpt_dir / step_name
    tmp_step = ckpt_dir / f".tmp_import_{step_name}"
    shutil.copy2(src, tmp_step)
    os.replace(tmp_step, step_dst)

    cfg.to_yaml(ckpt_dir / "model_config.yaml")

    if tokenizer_source is not None:
        tok_src = Path(tokenizer_source)
        if not tok_src.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_src}")
        shutil.copy2(tok_src, ckpt_dir / "tokenizer.model")
        tok_cfg = tok_src.with_name("tokenizer_config.json")
        if tok_cfg.exists():
            shutil.copy2(tok_cfg, ckpt_dir / "tokenizer_config.json")

    history_payload = {
        "training_status": payload.get("training_status", "trained"),
        "training_stage": payload.get("training_stage", "pretraining"),
        "current_learning_rate": payload.get("current_learning_rate"),
        "global_step": global_step,
        "max_steps": int((payload.get("training_config") or {}).get("max_steps", 0)),
        "dataset_tokens": int(payload.get("dataset_tokens", 0)),
        "checkpoint_version": payload.get("checkpoint_version", f"step-{global_step:07d}"),
        "saved_at": payload.get("saved_at"),
        "history": payload.get("training_history", []),
    }
    with open(ckpt_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history_payload, fh, indent=2)

    return {
        "imported": True,
        "target_path": str(target_latest),
        "global_step": global_step,
        "training_status": history_payload["training_status"],
        "training_stage": history_payload["training_stage"],
        "parameter_count": param_count,
    }
