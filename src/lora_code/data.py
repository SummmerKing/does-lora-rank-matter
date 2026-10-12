"""Training data: formatting, response-only loss masking, decontamination.

Loss masking
------------
For a causal LM trained on (prompt, response) pairs, the objective should be

    L = - (1 / |R|) * sum_{t in R} log p(x_t | x_<t),

where R is the set of *response* token positions. Positions belonging to the
prompt (system message, user instruction, chat-template scaffolding) and to
padding get label -100, which PyTorch's cross-entropy ignores. Training on
prompt or padding tokens inflates the token count with easy-to-predict
positions and makes the loss uninformative about code quality.

The prompt and response are tokenized *separately* and concatenated. This
guarantees the prompt tokens are identical to what the model sees at
inference time (where the prompt is tokenized alone), so the prompt/response
boundary can never be shifted by a BPE merge across it.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

IGNORE_INDEX = -100

SYSTEM_PROMPT = "You are a helpful assistant that writes correct, self-contained Python code."


def build_messages(instruction: str, response: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction.strip()},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def prompt_text(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(instruction), tokenize=False, add_generation_prompt=True
    )


def encode_example(tokenizer, instruction: str, response: str, max_length: int) -> dict | None:
    """Return {"input_ids", "labels"} with the prompt masked, or None if the
    example does not fit in max_length (we drop rather than truncate, so the
    model never learns to stop mid-function)."""
    prompt = prompt_text(tokenizer, instruction)
    full = tokenizer.apply_chat_template(
        build_messages(instruction, response), tokenize=False, add_generation_prompt=False
    )
    if not full.startswith(prompt):
        raise ValueError("chat template is not prefix-consistent; cannot locate the response boundary")
    completion = full[len(prompt):]  # response + end-of-turn token(s)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(completion_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


@dataclass
class Collator:
    """Right-pads a batch. Padding positions get attention 0 and label -100."""

    pad_token_id: int

    def __call__(self, features: Sequence[dict]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            mask.append([1] * n + [0] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------- normalising

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def normalise_response(output: str) -> str | None:
    """Training targets are a single fenced Python block, so the trained model
    and the evaluation code extractor agree on format. Returns None for
    responses whose code does not parse."""
    output = output.strip()
    m = FENCE_RE.search(output)
    code = (m.group(1) if m else output).strip()
    if not code or not is_valid_python(code):
        return None
    return f"```python\n{code}\n```"


def extract_fields(example: dict) -> tuple[str, str]:
    """Map common instruction-dataset schemas to (instruction, output)."""
    instruction = example.get("instruction") or example.get("prompt") or example.get("question") or ""
    extra = example.get("input") or ""
    if isinstance(extra, str) and extra.strip():
        instruction = f"{instruction.strip()}\n\nInput:\n{extra.strip()}"
    output = example.get("output") or example.get("completion") or example.get("response") or ""
    return instruction, output


# ------------------------------------------------------------ decontamination

def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _norm_tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_ngram_index(texts: Iterable[str], n: int = 13) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for t in texts:
        index |= ngrams(t, n)
    return index


def is_contaminated(text: str, index: set[tuple[str, ...]], n: int = 13) -> bool:
    """True if `text` shares any n-gram with the benchmark index (the
    13-gram overlap test used for GPT-3 contamination analysis)."""
    return not ngrams(text, n).isdisjoint(index)


# -------------------------------------------------------------- dataset build

def prepare_examples(
    rows: Iterable[dict],
    tokenizer,
    max_length: int,
    contamination_index: set | None = None,
    ngram_n: int = 13,
) -> tuple[list[dict], dict[str, int]]:
    stats = {"seen": 0, "kept": 0, "invalid": 0, "too_long": 0, "contaminated": 0, "duplicate": 0}
    seen_pairs = set()
    out = []
    for row in rows:
        stats["seen"] += 1
        instruction, output = extract_fields(row)
        response = normalise_response(output) if instruction.strip() else None
        if response is None:
            stats["invalid"] += 1
            continue
        key = (instruction.strip(), response)
        if key in seen_pairs:
            stats["duplicate"] += 1
            continue
        seen_pairs.add(key)
        if contamination_index is not None and is_contaminated(
            instruction + "\n" + response, contamination_index, ngram_n
        ):
            stats["contaminated"] += 1
            continue
        enc = encode_example(tokenizer, instruction, response, max_length)
        if enc is None:
            stats["too_long"] += 1
            continue
        out.append(enc)
        stats["kept"] += 1
    return out, stats


def split_train_val(examples: list, val_size: int, seed: int = 1234) -> tuple[list, list]:
    """Fixed-seed split, independent of the training seed, so every run in the
    study sees exactly the same train and validation sets."""
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    val = [examples[i] for i in idx[:val_size]]
    train = [examples[i] for i in idx[val_size:]]
    return train, val
