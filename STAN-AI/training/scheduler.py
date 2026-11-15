from __future__ import annotations

import math
from typing import Any

import torch


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay down to min_lr, with full state serialization."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_step = 0
        self._apply_lr(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        denom = max(1, self.max_steps - self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.max_lr - self.min_lr)

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        self._apply_lr(lr)
        return lr

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.warmup_steps = int(state["warmup_steps"])
        self.max_steps = int(state["max_steps"])
        self.max_lr = float(state["max_lr"])
        self.min_lr = float(state["min_lr"])
        self.last_step = int(state["last_step"])
        self._apply_lr(self.get_lr(self.last_step))


def create_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> CosineWarmupScheduler:
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )
