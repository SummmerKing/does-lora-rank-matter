"""Build the decontaminated, tokenized train/validation sets.

Writes data/train.jsonl, data/val.jsonl and data/stats.json.
"""

import json

from _common import DATA_DIR, GRID, write_jsonl
from datasets import load_dataset
from transformers import AutoTokenizer

from lora_code.benchmarks import benchmark_texts, load_benchmark
from lora_code.data import build_ngram_index, prepare_examples, split_train_val


def main():
    tok = AutoTokenizer.from_pretrained(GRID["model_name"])
    bench = [t for name in ("humaneval", "mbpp") for t in benchmark_texts(load_benchmark(name))]
    index = build_ngram_index(bench, n=13)

    rows = load_dataset(GRID["train_dataset"], split="train")
    examples, stats = prepare_examples(rows, tok, GRID["max_length"], contamination_index=index)
    train, val = split_train_val(examples, GRID["val_size"])
    stats.update(train=len(train), val=len(val))

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
