"""Training loop.

With gradient accumulation the loss is summed over the whole window and divided
by the number of response tokens in it, so examples are weighted per token
rather than per micro-batch.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE_INDEX, Collator
from .lora import count_parameters, lora_state_dict
from .optim import build_optimizer, build_scheduler


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    rslora: bool = False
    loraplus_ratio: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: float = 1.0
    micro_batch_size: int = 8
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_every: int = 50
    seed: int = 0
    bf16: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def summed_token_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Return (sum of next-token NLL over label positions, number of such positions)."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = batch["labels"][:, 1:]
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    n_tokens = int((shift_labels != IGNORE_INDEX).sum())
    return loss_sum, n_tokens


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _autocast(device: torch.device, bf16: bool):
    enabled = bf16 and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def evaluate_loss(model, examples, collate, batch_size: int, device, bf16: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = _to(collate(examples[i:i + batch_size]), device)
        with _autocast(device, bf16):
            s, n = summed_token_loss(model, batch)
        total += s.item()
        count += n
    model.train()
    return total / max(count, 1)


def micro_batches(examples: list, size: int, seed: int, epoch: int) -> list[list]:
    idx = list(range(len(examples)))
    random.Random(seed * 100_003 + epoch).shuffle(idx)
    return [[examples[j] for j in idx[i:i + size]] for i in range(0, len(idx), size)]


def train(model, train_examples: list, val_examples: list, pad_token_id: int, cfg: TrainConfig,
          out_dir: str | Path, log=print) -> dict:
    """Train the (already LoRA-wrapped) model. Returns the metrics dict that is
    also written to out_dir/metrics.json, alongside adapter.pt and config.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    collate = Collator(pad_token_id)
    set_seed(cfg.seed)

    steps_per_epoch = math.ceil(len(train_examples) / cfg.micro_batch_size / cfg.grad_accum)
    total_steps = max(1, round(steps_per_epoch * cfg.epochs))
    warmup = round(cfg.warmup_ratio * total_steps)

    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.loraplus_ratio)
    scheduler = build_scheduler(optimizer, total_steps, warmup)

    history = {"step": [], "train_loss": [], "lr": [], "val_step": [], "val_loss": []}
    params = count_parameters(model)
    log(f"trainable {params['trainable']:,} / {params['total']:,} params; {total_steps} optimizer steps")

    val0 = evaluate_loss(model, val_examples, collate, cfg.micro_batch_size, device, cfg.bf16)
    history["val_step"].append(0)
    history["val_loss"].append(val0)
    log(f"step 0 | val_loss {val0:.4f}")

    model.train()
    stream, epoch, step, t0 = [], 0, 0, time.time()
    while step < total_steps:
        while len(stream) < cfg.grad_accum:
            stream += micro_batches(train_examples, cfg.micro_batch_size, cfg.seed, epoch)
            epoch += 1
        window = [collate(mb) for mb in stream[:cfg.grad_accum]]
        stream = stream[cfg.grad_accum:]

        # Count response tokens in the whole window first, so every micro-batch
        # is divided by the same denominator (exact token-level mean).
        n_window = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in window)
        loss_value = 0.0
        for b in window:
            b = _to(b, device)
            with _autocast(device, cfg.bf16):
                s, _ = summed_token_loss(model, b)
            (s / n_window).backward()
            loss_value += s.item() / n_window

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        history["step"].append(step)
        history["train_loss"].append(loss_value)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if step % cfg.eval_every == 0 or step == total_steps:
            v = evaluate_loss(model, val_examples, collate, cfg.micro_batch_size, device, cfg.bf16)
            history["val_step"].append(step)
            history["val_loss"].append(v)
            log(f"step {step}/{total_steps} | train {loss_value:.4f} | val {v:.4f} | {time.time() - t0:.0f}s")

    torch.save(lora_state_dict(model), out_dir / "adapter.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = {
        "final_val_loss": history["val_loss"][-1],
        "initial_val_loss": val0,
        "total_steps": total_steps,
        "trainable_params": params["trainable"],
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
