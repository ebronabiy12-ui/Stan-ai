from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Evaluate validation cross-entropy loss and perplexity without gradient updates."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0

    amp_enabled = use_amp and device.type == "cuda"
    for idx, (x, y) in enumerate(val_loader):
        if idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            _, loss = model(x, targets=y)
        if loss is not None:
            total_loss += float(loss.item())
            batches += 1

    if was_training:
        model.train()

    if batches == 0:
        return {"val_loss": None, "perplexity": None, "batches_evaluated": 0}

    avg_loss = total_loss / batches
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_loss": avg_loss, "perplexity": ppl, "batches_evaluated": batches}
