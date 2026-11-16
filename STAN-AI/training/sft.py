from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
import yaml

from data.sft_dataset import SFTDataset, load_and_split_sft_jsonl
from model.config import STANConfig
from model.transformer import STANTransformer
from tokenizer.tokenizer import STANTokenizer
from training.checkpoint import (
    load_checkpoint,
    resolve_latest_checkpoint,
    save_checkpoint,
    validate_checkpoint_overwrite,
)
from training.optimizer import create_adamw_optimizer
from training.scheduler import create_cosine_warmup_scheduler
from training.train import set_reproducible_seed


@dataclass
class SFTConfig:
    seed: int = 42
    batch_size: int = 8
    grad_accum_steps: int = 4
    seq_len: int = 512
    max_steps: int = 2000
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    warmup_steps: int = 100
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: bool = True
    eval_interval: int = 100
    eval_steps: int = 25
    save_interval: int = 250
    log_interval: int = 25
    sft_data_path: str = "STAN-AI/data/processed/sft_instructions.jsonl"
    tokenizer_path: str = "STAN-AI/checkpoints/tokenizer.model"
    pretrained_checkpoint: str = "STAN-AI/checkpoints/latest.pt"
    checkpoint_dir: str = "STAN-AI/checkpoints"
    val_ratio: float = 0.1
    training_status_on_complete: str = "instruction-tuned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in valid})


@torch.no_grad()
def evaluate_sft_model(
    model: STANTransformer,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
    use_amp: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    amp_enabled = use_amp and device.type == "cuda"

    for idx, (x, y, attn) in enumerate(val_loader):
        if idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            _, loss = model(x, targets=y, attention_mask=attn, ignore_index=-100)
        if loss is not None and not torch.isnan(loss):
            total_loss += float(loss.item())
            batches += 1

    if was_training:
        model.train()

    if batches == 0:
        return {"val_loss": None, "perplexity": None, "batches_evaluated": 0}
    avg = total_loss / batches
    return {"val_loss": avg, "perplexity": math.exp(min(avg, 20.0)), "batches_evaluated": batches}


def run_instruction_tuning(
    sft_config: SFTConfig,
    resume_from: Optional[str | Path] = None,
    device_override: Optional[str] = None,
    overwrite_existing: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """
    Execute Stage 2 Supervised Instruction Tuning (SFT).
    Strictly requires a completed pretrained checkpoint (status in {'trained', 'instruction-tuned'})
    and never starts instruction tuning from random initialization or synthetic smoke-test weights.
    """
    set_reproducible_seed(sft_config.seed)

    base_ckpt_path = Path(sft_config.pretrained_checkpoint)
    if resume_from is not None:
        ckpt_to_load = (
            resolve_latest_checkpoint(sft_config.checkpoint_dir)
            if str(resume_from) == "auto"
            else Path(resume_from)
        )
        if ckpt_to_load is None or not ckpt_to_load.exists():
            raise FileNotFoundError("No checkpoint found to resume Stage 2 instruction tuning.")
    else:
        if not base_ckpt_path.exists():
            raise FileNotFoundError(
                f"Stage 2 requires a completed pretrained checkpoint at {base_ckpt_path}. "
                "Never start instruction tuning from random initialization."
            )
        validate_checkpoint_overwrite(
            sft_config.checkpoint_dir,
            overwrite_existing=overwrite_existing,
            resume_from=resume_from,
        )
        ckpt_to_load = base_ckpt_path

    raw_payload = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    base_status = str(raw_payload.get("training_status", "untrained"))
    if base_status not in {"trained", "instruction-tuned"}:
        raise ValueError(
            f"Cannot run Stage 2 instruction tuning from checkpoint with status '{base_status}'. "
            "A completed Stage 1 pretrained checkpoint ('trained') is required."
        )

    device = (
        torch.device(device_override)
        if device_override
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_cfg = STANConfig.from_dict(raw_payload["model_config"])
    tokenizer = STANTokenizer(sft_config.tokenizer_path)

    train_records, val_records = load_and_split_sft_jsonl(
        sft_config.sft_data_path,
        val_ratio=sft_config.val_ratio,
        seed=sft_config.seed,
    )
    train_ds = SFTDataset(train_records, tokenizer=tokenizer, max_seq_len=model_cfg.max_seq_len)
    val_ds = SFTDataset(val_records, tokenizer=tokenizer, max_seq_len=model_cfg.max_seq_len)
    dataset_tokens = train_ds.total_supervised_tokens + val_ds.total_supervised_tokens

    train_loader = DataLoader(
        train_ds,
        batch_size=sft_config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=sft_config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = STANTransformer(model_cfg, log_params=True).to(device)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        beta1=sft_config.beta1,
        beta2=sft_config.beta2,
    )
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=sft_config.warmup_steps,
        max_steps=sft_config.max_steps,
        max_lr=sft_config.learning_rate,
        min_lr=sft_config.min_learning_rate,
    )

    use_amp = bool(sft_config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    is_resuming_sft = (
        resume_from is not None and raw_payload.get("training_stage") == "instruction_tuning"
    )
    if is_resuming_sft:
        payload = load_checkpoint(
            ckpt_to_load,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_step = int(payload.get("global_step", 0))
        history = list(payload.get("training_history", []))
    else:
        # Fresh SFT optimizer/scheduler starting from pretrained Stage 1 weights
        load_checkpoint(ckpt_to_load, model=model, map_location=device)
        start_step = 0
        history = []

    model.train()
    train_iter = iter(train_loader)
    global_step = start_step
    last_train_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    last_lr: float = sft_config.learning_rate
    stopped_early = False

    while global_step < sft_config.max_steps:
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            save_checkpoint(
                checkpoint_dir=sft_config.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                model_config=model_cfg,
                training_config=sft_config.to_dict(),
                training_history=history,
                training_status="training",
                tokenizer_path=sft_config.tokenizer_path,
                dataset_tokens=dataset_tokens,
                training_stage="instruction_tuning",
                current_learning_rate=last_lr,
            )
            break

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(sft_config.grad_accum_steps):
            try:
                x, y, attn = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, attn = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(x, targets=y, attention_mask=attn, ignore_index=-100)
                assert loss is not None
                scaled_loss = loss / float(sft_config.grad_accum_steps)

            if scaler is not None and use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += float(loss.item()) / float(sft_config.grad_accum_steps)

        if scaler is not None and use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)
            optimizer.step()

        last_lr = scheduler.step()
        global_step += 1
        last_train_loss = accum_loss

        should_eval = (
            global_step % sft_config.eval_interval == 0
            or global_step == sft_config.max_steps
        )
        val_loss_val: Optional[float] = None
        if should_eval:
            ev = evaluate_sft_model(
                model,
                val_loader,
                device=device,
                max_batches=sft_config.eval_steps,
                use_amp=use_amp,
            )
            val_loss_val = ev["val_loss"]
            if val_loss_val is not None:
                last_val_loss = val_loss_val

        if global_step % sft_config.log_interval == 0 or should_eval or global_step == 1:
            history.append(
                {
                    "step": global_step,
                    "train_loss": round(accum_loss, 6),
                    "val_loss": round(val_loss_val, 6) if val_loss_val is not None else None,
                    "learning_rate": last_lr,
                }
            )

        if global_step % sft_config.save_interval == 0 or global_step == sft_config.max_steps:
            status = (
                sft_config.training_status_on_complete
                if global_step >= sft_config.max_steps
                else "training"
            )
            save_checkpoint(
                checkpoint_dir=sft_config.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                model_config=model_cfg,
                training_config=sft_config.to_dict(),
                training_history=history,
                training_status=status,
                tokenizer_path=sft_config.tokenizer_path,
                dataset_tokens=dataset_tokens,
                training_stage="instruction_tuning",
                current_learning_rate=last_lr,
            )

    return {
        "global_step": global_step,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "current_learning_rate": last_lr,
        "dataset_tokens": dataset_tokens,
        "checkpoint_dir": sft_config.checkpoint_dir,
        "history": history,
        "status": (
            sft_config.training_status_on_complete
            if global_step >= sft_config.max_steps
            else "training"
        ),
        "training_stage": "instruction_tuning",
        "stopped_early": stopped_early,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Instruction Tuning for STAN AI.")
    parser.add_argument("--sft-config", default="STAN-AI/configs/sft.yaml")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--sft-data", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = SFTConfig.from_yaml(args.sft_config)
    if args.pretrained_checkpoint:
        cfg.pretrained_checkpoint = args.pretrained_checkpoint
    if args.sft_data:
        cfg.sft_data_path = args.sft_data
    if args.tokenizer:
        cfg.tokenizer_path = args.tokenizer
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    summary = run_instruction_tuning(
        sft_config=cfg,
        resume_from=args.resume,
        device_override=args.device,
        overwrite_existing=args.overwrite,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
