"""Optimizer and learning-rate schedule.

LoRA+ (Hayou, Ghosh & Yu, 2024) argues that A and B should not share a
learning rate: in the infinite-width limit, efficient feature learning
needs lr_B / lr_A = Theta(n). In practice they suggest lr_B = lambda * lr_A
with lambda around 16. With lambda = 1 this reduces to plain LoRA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    loraplus_ratio: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    params_a, params_b = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("lora_A"):
            params_a.append(p)
        elif name.endswith("lora_B"):
            params_b.append(p)
        else:
            raise ValueError(f"unexpected trainable parameter {name}; only LoRA adapters should train")
    if not params_a:
        raise ValueError("no LoRA parameters found")
    groups = [
        {"params": params_a, "lr": lr, "name": "lora_A"},
        {"params": params_b, "lr": lr * loraplus_ratio, "name": "lora_B"},
    ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=betas)


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.0) -> float:
    """LR multiplier: linear warmup from 1/warmup to 1, then cosine decay to min_ratio.

    Step indices are 0-based, so the first optimizer step uses a non-zero LR.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.0):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_cosine(s, total_steps, warmup_steps, min_ratio)
    )
