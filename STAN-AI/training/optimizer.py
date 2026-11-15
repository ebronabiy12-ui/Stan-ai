from __future__ import annotations

import torch
import torch.nn as nn


def create_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with decoupled weight decay applied only to 2D+ weight tensors
    (excluding 1D norm scales and bias vectors). Tied parameters are deduplicated.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_ids: set[int] = set()

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
    )
