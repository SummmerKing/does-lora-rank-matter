"""End-to-end smoke test on CPU with a tiny random model: training must reduce
the response-only loss, touch only adapter weights, and save a loadable adapter."""

import json

import torch

from lora_code.data import encode_example
from lora_code.lora import apply_lora, load_lora_state_dict, lora_layers
from lora_code.train import TrainConfig, train

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_training_reduces_loss_and_saves_adapter(tokenizer, tiny_model, tmp_path):
    pairs = [(f"Return the number {i}.", f"```python\ndef f():\n    return {i}\n```") for i in range(16)]
    examples = [encode_example(tokenizer, a, b, 256) for a, b in pairs]
    frozen = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}

    apply_lora(tiny_model, TARGETS, rank=8, alpha=16)
    cfg = TrainConfig(rank=8, alpha=16, dropout=0.0, lr=1e-2, epochs=8, micro_batch_size=4, grad_accum=2,
                      warmup_ratio=0.1, eval_every=1000, seed=0, bf16=False, target_modules=TARGETS)
    m = train(tiny_model, examples, examples[:8], tokenizer.pad_token_id, cfg, tmp_path, log=lambda *_: None)

    assert m["final_val_loss"] < 0.5 * m["initial_val_loss"]
    for name, p in tiny_model.named_parameters():
        if "lora_" not in name:
            key = name.replace(".base.", ".")
            assert torch.equal(p, frozen[key]), f"base weight {name} changed"

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["total_steps"] == 16 and len(saved["history"]["train_loss"]) == 16
    state = torch.load(tmp_path / "adapter.pt")
    for layer in lora_layers(tiny_model):
        torch.nn.init.zeros_(layer.lora_B)
    load_lora_state_dict(tiny_model, state)
    assert any(layer.lora_B.abs().sum() > 0 for layer in lora_layers(tiny_model))


def test_training_is_deterministic(tokenizer, tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    def run(out):
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
        apply_lora(model, TARGETS, rank=4, alpha=8)
        ex = [encode_example(tokenizer, f"q{i}", f"```python\nx = {i}\n```", 128) for i in range(8)]
        cfg = TrainConfig(rank=4, alpha=8, dropout=0.0, lr=1e-3, epochs=1, micro_batch_size=2, grad_accum=2,
                          seed=7, bf16=False, target_modules=TARGETS)
        return train(model, ex, ex[:2], 0, cfg, out, log=lambda *_: None)["history"]["train_loss"]

    assert run(tmp_path / "a") == run(tmp_path / "b")
