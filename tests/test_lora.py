import math

import pytest
import torch
import torch.nn as nn

from lora_code.lora import (LoRALinear, apply_lora, count_parameters, expected_lora_params,
                            load_lora_state_dict, lora_layers, lora_state_dict, merge_all)
from lora_code.optim import build_optimizer, warmup_cosine

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_zero_init_is_identity():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    expected = base(x)
    layer = LoRALinear(base, rank=4, alpha=8)
    assert torch.allclose(layer(x), expected)


def test_forward_matches_dense_formula():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, rank=4, alpha=8)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(5, 16)
    w = base.weight + (8 / 4) * layer.lora_B @ layer.lora_A
    assert torch.allclose(layer(x), x @ w.T + base.bias, atol=1e-5)


def test_rslora_scaling():
    layer = LoRALinear(nn.Linear(4, 4), rank=16, alpha=32, rslora=True)
    assert layer.scaling == pytest.approx(32 / math.sqrt(16))


def test_merge_and_unmerge_round_trip():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    w0 = base.weight.clone()
    layer = LoRALinear(base, rank=4, alpha=8)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(3, 16)
    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)  # same function, one matmul
    layer.unmerge()
    assert torch.allclose(base.weight, w0, atol=1e-6)
    assert torch.allclose(layer(x), before, atol=1e-5)


def test_gradients_only_reach_adapters_and_match_closed_form():
    torch.manual_seed(0)
    base = nn.Linear(6, 3, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(5, 6)
    g = torch.randn(5, 3)  # upstream gradient dL/dy
    (layer(x) * g).sum().backward()
    assert base.weight.grad is None
    s = layer.scaling
    a, b = layer.lora_A.detach(), layer.lora_B.detach()
    # y = s * x A^T B^T  =>  dL/dB = s * g^T (x A^T),  dL/dA = s * B^T g^T x
    assert torch.allclose(layer.lora_B.grad, s * g.T @ (x @ a.T), atol=1e-5)
    assert torch.allclose(layer.lora_A.grad, s * b.T @ g.T @ x, atol=1e-5)


def test_apply_lora_exact_name_match_and_param_count(tiny_model):
    shapes = [(m.out_features, m.in_features) for n, m in tiny_model.named_modules()
              if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in TARGETS]
    wrapped = apply_lora(tiny_model, TARGETS, rank=4, alpha=8)
    assert len(wrapped) == len(shapes) == 2 * 7
    assert all(n.rsplit(".", 1)[-1] in TARGETS for n in wrapped)
    assert "lm_head" not in " ".join(wrapped)
    assert count_parameters(tiny_model)["trainable"] == expected_lora_params(shapes, 4)


def test_wrapped_model_unchanged_at_init(tiny_model):
    ids = torch.randint(0, 512, (2, 10))
    before = tiny_model(input_ids=ids).logits
    apply_lora(tiny_model, TARGETS, rank=4, alpha=8)
    assert torch.allclose(tiny_model(input_ids=ids).logits, before, atol=1e-6)


def test_adapter_save_load_and_merge(tiny_model):
    apply_lora(tiny_model, TARGETS, rank=4, alpha=8)
    for layer in lora_layers(tiny_model):
        nn.init.normal_(layer.lora_B, std=0.05)
    ids = torch.randint(0, 512, (2, 10))
    ref = tiny_model(input_ids=ids).logits
    state = lora_state_dict(tiny_model)
    for layer in lora_layers(tiny_model):
        nn.init.zeros_(layer.lora_B)
    load_lora_state_dict(tiny_model, state)
    assert torch.allclose(tiny_model(input_ids=ids).logits, ref, atol=1e-5)
    merge_all(tiny_model)
    assert torch.allclose(tiny_model(input_ids=ids).logits, ref, atol=1e-4)


def test_loraplus_learning_rates(tiny_model):
    apply_lora(tiny_model, TARGETS, rank=4, alpha=8)
    opt = build_optimizer(tiny_model, lr=1e-4, loraplus_ratio=16)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs == {"lora_A": 1e-4, "lora_B": pytest.approx(1.6e-3)}
    assert sum(len(g["params"]) for g in opt.param_groups) == 2 * 14


def test_optimizer_rejects_non_adapter_params(tiny_model):
    apply_lora(tiny_model, TARGETS, rank=4, alpha=8)
    tiny_model.lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        build_optimizer(tiny_model, lr=1e-4)


def test_warmup_cosine_schedule():
    total, warm = 100, 10
    assert warmup_cosine(0, total, warm) == pytest.approx(0.1)
    assert warmup_cosine(9, total, warm) == pytest.approx(1.0)
    assert warmup_cosine(10, total, warm) == pytest.approx(1.0)
    assert warmup_cosine(55, total, warm) == pytest.approx(0.5)  # half-way through the decay
    assert warmup_cosine(100, total, warm) == pytest.approx(0.0, abs=1e-12)
    values = [warmup_cosine(s, total, warm) for s in range(warm, total + 1)]
    assert all(a >= b for a, b in zip(values, values[1:]))
