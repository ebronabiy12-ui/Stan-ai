from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from model.config import STANConfig
from model.transformer import STANTransformer
from tokenizer.tokenizer import STANTokenizer
from training.checkpoint import load_checkpoint, resolve_latest_checkpoint

PRODUCTION_READY_STATES = {"trained", "instruction-tuned"}


@dataclass
class LoadedModelState:
    model: Optional[STANTransformer]
    model_config: STANConfig
    tokenizer: Optional[STANTokenizer]
    device: torch.device
    training_status: str  # untrained | training | trained | instruction-tuned | smoke_test
    checkpoint_loaded: bool
    checkpoint_path: Optional[str]
    checkpoint_version: Optional[str]
    global_step: int
    max_steps: int
    dataset_tokens: int
    parameter_breakdown: dict[str, int]
    training_history: list[dict[str, Any]] = field(default_factory=list)
    warning: Optional[str] = None
    saved_at: Optional[str] = None
    training_stage: str = "untrained"
    current_learning_rate: Optional[float] = None

    @property
    def production_chat_enabled(self) -> bool:
        return (
            self.checkpoint_loaded
            and self.model is not None
            and self.tokenizer is not None
            and self.training_status in PRODUCTION_READY_STATES
        )


def _select_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_untrained_parameter_breakdown(cfg: STANConfig) -> dict[str, int]:
    # Instantiate lightweight meta/CPU count directly from architecture module
    model = STANTransformer(cfg, log_params=False)
    breakdown = model.parameter_breakdown()
    del model
    return breakdown


def load_stan_runtime(
    model_config_path: str | Path = "STAN-AI/configs/model.yaml",
    checkpoint_dir: str | Path = "STAN-AI/checkpoints",
    tokenizer_path: Optional[str | Path] = None,
    device_override: Optional[str] = None,
) -> LoadedModelState:
    """
    Load STAN AI checkpoint and tokenizer once for inference.
    Strictly distinguishes untrained, training, smoke_test, trained, and instruction-tuned states.
    """
    device = _select_device(device_override)
    cfg_path = Path(model_config_path)
    ckpt_dir = Path(checkpoint_dir)

    base_config = STANConfig.from_yaml(cfg_path) if cfg_path.exists() else STANConfig()

    ckpt_file = resolve_latest_checkpoint(ckpt_dir)
    history_file = ckpt_dir / "training_history.json"
    history_data: list[dict[str, Any]] = []
    max_steps = 0
    dataset_tokens = 0
    saved_at: Optional[str] = None
    hist_stage = "untrained"
    hist_lr: Optional[float] = None

    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as fh:
            parsed_hist = json.load(fh)
        history_data = list(parsed_hist.get("history", []))
        max_steps = int(parsed_hist.get("max_steps", 0))
        dataset_tokens = int(parsed_hist.get("dataset_tokens", 0))
        saved_at = parsed_hist.get("saved_at")
        hist_stage = str(parsed_hist.get("training_stage", "untrained"))
        hist_lr = parsed_hist.get("current_learning_rate")

    # Resolve tokenizer if present
    tok_candidate = (
        Path(tokenizer_path)
        if tokenizer_path is not None
        else ckpt_dir / "tokenizer.model"
    )
    tokenizer: Optional[STANTokenizer] = None
    if tok_candidate.exists():
        tokenizer = STANTokenizer(tok_candidate)

    if ckpt_file is None or not ckpt_file.exists():
        breakdown = _compute_untrained_parameter_breakdown(base_config)
        return LoadedModelState(
            model=None,
            model_config=base_config,
            tokenizer=tokenizer,
            device=device,
            training_status="untrained",
            checkpoint_loaded=False,
            checkpoint_path=None,
            checkpoint_version=None,
            global_step=0,
            max_steps=max_steps,
            dataset_tokens=dataset_tokens,
            parameter_breakdown=breakdown,
            training_history=history_data,
            warning=(
                "No trained checkpoint found in checkpoints directory. "
                "STAN AI has not been trained yet; production chat is disabled."
            ),
            saved_at=saved_at,
            training_stage=hist_stage,
            current_learning_rate=hist_lr,
        )

    # Peek payload config before building model so architecture matches checkpoint
    raw_payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    ckpt_cfg_dict = raw_payload.get("model_config")
    model_cfg = STANConfig.from_dict(ckpt_cfg_dict) if ckpt_cfg_dict else base_config

    model = STANTransformer(model_cfg, log_params=True).to(device)
    payload = load_checkpoint(ckpt_file, model=model, map_location=device)
    model.eval()

    training_status = str(payload.get("training_status", "untrained"))
    training_stage = str(payload.get("training_stage", hist_stage if hist_stage != "untrained" else "pretraining"))
    current_lr = payload.get("current_learning_rate", hist_lr)
    global_step = int(payload.get("global_step", 0))
    ckpt_version = str(payload.get("checkpoint_version", f"step-{global_step:07d}"))
    ckpt_history = list(payload.get("training_history", history_data))
    if current_lr is None and ckpt_history:
        current_lr = ckpt_history[-1].get("learning_rate")
    ckpt_train_cfg = payload.get("training_config") or {}
    max_steps = int(ckpt_train_cfg.get("max_steps", max_steps))
    dataset_tokens = int(payload.get("dataset_tokens", dataset_tokens))
    saved_at = payload.get("saved_at", saved_at)

    warning: Optional[str] = None
    if training_status == "smoke_test":
        warning = (
            "Loaded checkpoint was trained on the synthetic numerical smoke-test dataset, "
            "not an English language corpus. Production chat is disabled."
        )
    elif training_status not in PRODUCTION_READY_STATES:
        warning = (
            f"Checkpoint status is '{training_status}' (step {global_step}). "
            "Production chat is enabled only when status is 'trained' or 'instruction-tuned'."
        )
    elif tokenizer is None:
        warning = (
            "Checkpoint weights loaded, but custom SentencePiece tokenizer.model is missing. "
            "Text generation and tokenization are unavailable until tokenizer.model is present."
        )

    return LoadedModelState(
        model=model,
        model_config=model_cfg,
        tokenizer=tokenizer,
        device=device,
        training_status=training_status,
        checkpoint_loaded=True,
        checkpoint_path=str(ckpt_file),
        checkpoint_version=ckpt_version,
        global_step=global_step,
        max_steps=max_steps,
        dataset_tokens=dataset_tokens,
        parameter_breakdown=model.parameter_breakdown(),
        training_history=ckpt_history,
        warning=warning,
        saved_at=saved_at,
        training_stage=training_stage,
        current_learning_rate=current_lr,
    )


class STANModelRegistry:
    """Singleton-style holder ensuring weights are loaded once per FastAPI process."""

    def __init__(
        self,
        model_config_path: Optional[str | Path] = None,
        checkpoint_dir: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
        device_override: Optional[str] = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[1]
        self.model_config_path = Path(
            model_config_path
            or os.environ.get("STAN_MODEL_CONFIG", str(default_root / "configs" / "model.yaml"))
        )
        self.checkpoint_dir = Path(
            checkpoint_dir
            or os.environ.get("STAN_CHECKPOINT_DIR", str(default_root / "checkpoints"))
        )
        self.tokenizer_path = tokenizer_path or os.environ.get("STAN_TOKENIZER_PATH")
        self.device_override = device_override or os.environ.get("STAN_DEVICE")
        self._state: Optional[LoadedModelState] = None

    def get_state(self) -> LoadedModelState:
        if self._state is None:
            self._state = load_stan_runtime(
                model_config_path=self.model_config_path,
                checkpoint_dir=self.checkpoint_dir,
                tokenizer_path=self.tokenizer_path,
                device_override=self.device_override,
            )
        return self._state

    def reload(self) -> LoadedModelState:
        self._state = load_stan_runtime(
            model_config_path=self.model_config_path,
            checkpoint_dir=self.checkpoint_dir,
            tokenizer_path=self.tokenizer_path,
            device_override=self.device_override,
        )
        return self._state
