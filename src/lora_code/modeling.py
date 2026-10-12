"""Model loading and batched greedy generation."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from .data import prompt_text
from .lora import apply_lora, load_lora_state_dict, merge_all


def load_base(model_name: str, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # transformers renamed `torch_dtype` to `dtype`; support both.
    kw = "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_name, **{kw: dtype}).to(device)
    model.config.use_cache = False
    return model, tok


def load_finetuned(model_name: str, run_dir: str | Path, device: str = "cuda", merge: bool = True):
    import json

    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    model, tok = load_base(model_name, device)
    apply_lora(model, cfg["target_modules"], cfg["rank"], cfg["alpha"], 0.0, cfg.get("rslora", False))
    load_lora_state_dict(model, torch.load(Path(run_dir) / "adapter.pt", map_location=device))
    if merge:
        merge_all(model)
    return model, tok


@torch.no_grad()
def generate(model, tokenizer, instructions: list[str], max_new_tokens: int = 512,
             batch_size: int = 16) -> list[str]:
    """Greedy decoding with left padding. No repetition penalties: code
    legitimately repeats n-grams, and penalties are a confound."""
    model.eval()
    model.config.use_cache = True
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    outputs = []
    try:
        # Sort by length to reduce padding, then restore the original order.
        prompts = [prompt_text(tokenizer, ins) for ins in instructions]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        result = [None] * len(prompts)
        for i in range(0, len(order), batch_size):
            ids = order[i:i + batch_size]
            enc = tokenizer([prompts[j] for j in ids], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            for j, text in zip(ids, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
                result[j] = text
        outputs = result
    finally:
        tokenizer.padding_side = old_side
    return outputs
