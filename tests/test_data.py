import torch
import torch.nn.functional as F

from lora_code.data import (IGNORE_INDEX, Collator, build_ngram_index, encode_example, is_contaminated,
                            normalise_response, prepare_examples, split_train_val)
from lora_code.train import summed_token_loss


def test_only_response_tokens_are_labelled(tokenizer):
    ex = encode_example(tokenizer, "Add two numbers.", "```python\ndef add(a, b):\n    return a + b\n```", 512)
    labelled = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE_INDEX]
    n_prompt = ex["labels"].index(next(l for l in ex["labels"] if l != IGNORE_INDEX))
    assert all(l == IGNORE_INDEX for l in ex["labels"][:n_prompt])
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    text = tokenizer.decode(labelled)
    assert text.startswith("```python\ndef add")
    assert text.endswith("<|im_end|>\n")  # the model must learn to stop
    assert "Add two numbers" not in text and "system" not in text


def test_prompt_tokens_match_inference_prompt(tokenizer):
    from lora_code.data import prompt_text
    ex = encode_example(tokenizer, "Reverse a string.", "```python\ndef r(s):\n    return s[::-1]\n```", 512)
    prompt_ids = tokenizer(prompt_text(tokenizer, "Reverse a string."))["input_ids"]
    assert ex["input_ids"][:len(prompt_ids)] == prompt_ids
    assert ex["labels"][len(prompt_ids)] != IGNORE_INDEX


def test_too_long_examples_are_dropped_not_truncated(tokenizer):
    assert encode_example(tokenizer, "x", "```python\n" + "a = 1\n" * 200 + "```", 64) is None


def test_collator_masks_padding(tokenizer):
    a = encode_example(tokenizer, "a", "```python\nx = 1\n```", 512)
    b = encode_example(tokenizer, "a much longer instruction here", "```python\ny = [1, 2, 3]\n```", 512)
    batch = Collator(pad_token_id=0)([a, b])
    short = 0 if len(a["input_ids"]) < len(b["input_ids"]) else 1
    n = len([a, b][short]["input_ids"])
    assert (batch["labels"][short, n:] == IGNORE_INDEX).all()
    assert (batch["attention_mask"][short, n:] == 0).all()
    assert batch["attention_mask"][short, :n].all()


def test_loss_equals_manual_response_only_nll(tokenizer, tiny_model):
    ex = encode_example(tokenizer, "Square a number.", "```python\ndef sq(x):\n    return x * x\n```", 512)
    batch = Collator(0)([ex])
    loss_sum, n = summed_token_loss(tiny_model, batch)
    logits = tiny_model(input_ids=batch["input_ids"]).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    resp = [i for i, l in enumerate(ex["labels"]) if l != IGNORE_INDEX]
    manual = -sum(logp[i - 1, ex["input_ids"][i]] for i in resp)
    assert n == len(resp)
    assert torch.allclose(loss_sum, manual, atol=1e-4)


def test_padding_does_not_change_loss(tokenizer, tiny_model):
    a = encode_example(tokenizer, "a", "```python\nx = 1\n```", 512)
    b = encode_example(tokenizer, "a much longer instruction", "```python\ny = [1, 2, 3, 4]\n```", 512)
    alone, n_alone = summed_token_loss(tiny_model, Collator(0)([a]))
    both, n_both = summed_token_loss(tiny_model, Collator(0)([a, b]))
    b_alone, n_b = summed_token_loss(tiny_model, Collator(0)([b]))
    assert n_both == n_alone + n_b
    assert torch.allclose(both, alone + b_alone, atol=1e-3)


def test_normalise_response():
    assert normalise_response("def f():\n    return 1") == "```python\ndef f():\n    return 1\n```"
    assert normalise_response("Sure!\n```python\nx = 1\n```\nDone.") == "```python\nx = 1\n```"
    assert normalise_response("this is not code (") is None
    assert normalise_response("") is None


def test_decontamination():
    bench = ["def has_close_elements(numbers, threshold): check if in given list of numbers are any two "
             "numbers closer to each other than given threshold"]
    index = build_ngram_index(bench, n=13)
    leaked = "Write code: check if in given list of numbers are any two numbers closer to each other than given threshold"
    assert is_contaminated(leaked, index, 13)
    assert not is_contaminated("Write a function that sorts a list of strings by length", index, 13)


def test_prepare_examples_filters_and_dedups(tokenizer):
    rows = [
        {"instruction": "Add.", "input": "", "output": "def add(a, b): return a + b"},
        {"instruction": "Add.", "input": "", "output": "def add(a, b): return a + b"},  # duplicate
        {"instruction": "Bad.", "input": "", "output": "this is ( not python"},
        {"instruction": "", "input": "", "output": "x = 1"},
    ]
    examples, stats = prepare_examples(rows, tokenizer, 512)
    assert stats == {"seen": 4, "kept": 1, "invalid": 2, "too_long": 0, "contaminated": 0, "duplicate": 1}


def test_split_is_independent_of_training_seed():
    data = list(range(100))
    t1, v1 = split_train_val(data, 10)
    t2, v2 = split_train_val(data, 10)
    assert v1 == v2 and t1 == t2 and not set(v1) & set(t1) and len(t1) + len(v1) == 100
