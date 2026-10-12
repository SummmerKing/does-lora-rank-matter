import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ChatMLTokenizer:
    """Offline stand-in for a Qwen tokenizer: the same ChatML template and a
    deterministic word/punctuation vocabulary. Enough to test masking logic
    without downloading anything."""

    SPECIAL = ["<|pad|>", "<|im_start|>", "<|im_end|>"]
    PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\w+|\s|[^\w\s]")

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.SPECIAL)}
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"

    def _id(self, tok):
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]

    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False):
        if isinstance(text, str):
            return {"input_ids": [self._id(t) for t in self.PATTERN.findall(text)]}
        seqs = [self(t)["input_ids"] for t in text]
        width = max(map(len, seqs))
        ids, mask = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (width - len(s))
            m0, m1 = [0] * len(pad), [1] * len(s)
            ids.append(pad + s if self.padding_side == "left" else s + pad)
            mask.append(m0 + m1 if self.padding_side == "left" else m1 + m0)
        return _Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i >= len(self.SPECIAL)]
        return "".join(inv.get(i, "?") for i in ids)

    def batch_decode(self, rows, skip_special_tokens=False):
        return [self.decode(r, skip_special_tokens) for r in rows]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


@pytest.fixture
def tokenizer():
    return ChatMLTokenizer()


@pytest.fixture
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    # initializer_range=0.3: with the default 0.02 the frozen random lm_head is so
    # small that no adapter can produce confident logits, which would make the
    # training smoke test measure the toy model rather than the training code.
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      initializer_range=0.3)
    return Qwen2ForCausalLM(cfg).float()
