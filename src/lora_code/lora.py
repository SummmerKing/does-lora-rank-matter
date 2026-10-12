"""LoRA layers (Hu et al., 2021), written without peft.

    y = W0 x + s * B A x,   s = alpha / r   (alpha / sqrt(r) for rsLoRA)

B starts at zero, so the wrapped model is identical to the base model before
training. Adapters stay in fp32 while the base model runs in bf16.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        rslora: bool = False,
        adapter_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.rslora = rslora
        self.scaling = alpha / math.sqrt(rank) if rslora else alpha / rank
        self.merged = False

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=adapter_dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=adapter_dtype))
        # Same init as nn.Linear's weight, so A's scale matches the base layer.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """The dense update s * B A, shape (d_out, d_in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        h = self.dropout(x).to(self.lora_A.dtype)
        update = (h @ self.lora_A.T) @ self.lora_B.T
        return out + (update * self.scaling).to(out.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold s*BA into W0 for zero-overhead inference."""
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    rslora: bool = False,
) -> list[str]:
    """Freeze `model` and wrap every nn.Linear whose last name component
    is in `target_modules` (exact match, so "proj" never matches "o_proj").
    Returns the qualified names of the wrapped layers."""
    targets = set(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = []
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and isinstance(module, nn.Linear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, rank, alpha, dropout, rslora))
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no nn.Linear matched target_modules={sorted(targets)}")
    return wrapped


def lora_layers(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    # .clone(): on CPU, .cpu() is a no-op that shares storage with the live
    # weights, so later in-place updates would silently change the snapshot.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {k for k in model.state_dict() if "lora_A" in k or "lora_B" in k}
    if unexpected or (expected - set(state)):
        raise RuntimeError(f"adapter mismatch: unexpected={unexpected}, missing={sorted(expected - set(state))}")


def merge_all(model: nn.Module) -> None:
    for layer in lora_layers(model):
        layer.merge()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def expected_lora_params(shapes: Iterable[tuple[int, int]], rank: int) -> int:
    """Closed form: each wrapped (d_out, d_in) layer adds r * (d_in + d_out)."""
    return sum(rank * (d_in + d_out) for d_out, d_in in shapes)
