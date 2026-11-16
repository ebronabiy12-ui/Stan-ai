from __future__ import annotations

import os
import platform
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from data.dataset import generate_smoke_test_dataset
from inference.load_model import STANModelRegistry
from model.config import STANConfig
from training.sft import SFTConfig, run_instruction_tuning
from training.train import TrainingConfig, run_training


def get_hardware_info() -> dict[str, Any]:
    """Return honest, measured compute hardware and CUDA availability."""
    cuda_available = torch.cuda.is_available()
    gpus: list[dict[str, Any]] = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                }
            )

    total_ram_gb: Optional[float] = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_ram_gb = round(kb / (1024**2), 2)
                    break
    except OSError:
        pass

    return {
        "cuda_available": cuda_available,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "active_device": "cuda:0" if cuda_available else "cpu",
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": total_ram_gb,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "recommended_training_target": (
            "Local CUDA GPU ready for Stage 1 & Stage 2 training."
            if cuda_available
            else "No GPU detected on this host. Use Google Colab or Kaggle GPU notebook (STAN-AI/notebooks/STAN_AI_Colab_Kaggle_Training.ipynb) for full 51.66M English pretraining and import checkpoints back."
        ),
    }


class TrainingJobManager:
    """Thread-safe controller for launching, monitoring, resuming, and stopping training jobs."""

    def __init__(self, registry: STANModelRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state: dict[str, Any] = {
            "status": "idle",  # idle | running | stopping | completed | stopped | failed
            "stage": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "summary": None,
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start_job(
        self,
        stage: str,
        resume: bool = False,
        overwrite_existing: bool = False,
        max_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        if stage not in {"smoke_test", "pretraining", "instruction_tuning"}:
            raise ValueError(f"Unsupported training stage: {stage}")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    f"A training job ('{self._state['stage']}') is already running. Stop it first."
                )

            self._stop_event.clear()
            self._state = {
                "status": "running",
                "stage": stage,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "error": None,
                "summary": None,
            }

            def _runner() -> None:
                try:
                    if stage == "smoke_test":
                        # Smoke tests stay isolated in temp directory so production checkpoints are never polluted
                        work = Path(tempfile.gettempdir()) / "stan_smoke_verify"
                        smoke_meta = generate_smoke_test_dataset(
                            work / "data",
                            vocab_size=128,
                            train_tokens=2048,
                            val_tokens=512,
                        )
                        m_cfg = STANConfig(
                            model_name="STAN-AI-SmokeTest",
                            n_layers=2,
                            d_model=64,
                            n_heads=4,
                            d_head=16,
                            d_ff=128,
                            max_seq_len=32,
                            vocab_size=128,
                        )
                        t_cfg = TrainingConfig(
                            seed=42,
                            batch_size=4,
                            grad_accum_steps=1,
                            seq_len=32,
                            max_steps=max_steps or 20,
                            learning_rate=3e-3,
                            min_learning_rate=3e-4,
                            warmup_steps=4,
                            eval_interval=10,
                            eval_steps=4,
                            save_interval=10,
                            log_interval=5,
                            train_data_path=smoke_meta["train"]["bin_path"],
                            val_data_path=smoke_meta["val"]["bin_path"],
                            tokenizer_path=str(work / "none.model"),
                            checkpoint_dir=str(work / "checkpoints"),
                            training_status_on_complete="smoke_test",
                        )
                        summary = run_training(
                            model_config=m_cfg,
                            train_config=t_cfg,
                            resume_from="auto" if resume else None,
                            overwrite_existing=True,
                            stop_event=self._stop_event,
                            training_stage="smoke_test",
                        )
                    elif stage == "pretraining":
                        stan_root = Path(__file__).resolve().parents[1]
                        m_cfg = STANConfig.from_yaml(self.registry.model_config_path)
                        t_cfg = TrainingConfig.from_yaml(stan_root / "configs" / "training.yaml")
                        t_cfg.checkpoint_dir = str(self.registry.checkpoint_dir)
                        if max_steps is not None:
                            t_cfg.max_steps = max_steps
                        summary = run_training(
                            model_config=m_cfg,
                            train_config=t_cfg,
                            resume_from="auto" if resume else None,
                            overwrite_existing=overwrite_existing,
                            stop_event=self._stop_event,
                            training_stage="pretraining",
                        )
                        self.registry.reload()
                    else:
                        stan_root = Path(__file__).resolve().parents[1]
                        s_cfg = SFTConfig.from_yaml(stan_root / "configs" / "sft.yaml")
                        s_cfg.checkpoint_dir = str(self.registry.checkpoint_dir)
                        if max_steps is not None:
                            s_cfg.max_steps = max_steps
                        summary = run_instruction_tuning(
                            sft_config=s_cfg,
                            resume_from="auto" if resume else None,
                            overwrite_existing=overwrite_existing,
                            stop_event=self._stop_event,
                        )
                        self.registry.reload()

                    with self._lock:
                        self._state["status"] = "stopped" if summary.get("stopped_early") else "completed"
                        self._state["finished_at"] = datetime.now(timezone.utc).isoformat()
                        self._state["summary"] = {
                            k: v for k, v in summary.items() if k != "history"
                        }
                except Exception as exc:
                    with self._lock:
                        self._state["status"] = "failed"
                        self._state["finished_at"] = datetime.now(timezone.utc).isoformat()
                        self._state["error"] = str(exc)

            self._thread = threading.Thread(target=_runner, daemon=True)
            self._thread.start()
            return dict(self._state)

    def stop_job(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return dict(self._state)
            self._stop_event.set()
            self._state["status"] = "stopping"
            return dict(self._state)
