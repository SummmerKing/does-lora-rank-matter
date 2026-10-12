"""Train one configuration with one seed.

    python scripts/train.py --config lora_r16 --seed 0 [--lr 2e-4] [--epochs 1.0] [--out runs/...]

Without --lr, the learning rate chosen by scripts/sweep_lr.py is used.
"""

import argparse
import json

import torch
from _common import DATA_DIR, GRID, RUNS_DIR, read_jsonl

from lora_code.lora import apply_lora
from lora_code.modeling import load_base
from lora_code.train import TrainConfig, train


def selected_lr(config: str) -> float:
    path = RUNS_DIR / "selected_lr.json"
    if not path.exists():
        raise SystemExit("no --lr given and runs/selected_lr.json missing; run scripts/sweep_lr.py first")
    return json.loads(path.read_text())[config]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(GRID["configs"]))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--out")
    args = ap.parse_args()

    spec = {k: v for k, v in GRID["configs"][args.config].items() if k != "lr_grid"}
    cfg = TrainConfig(**GRID["common"], **spec, seed=args.seed,
                      lr=args.lr if args.lr is not None else selected_lr(args.config))
    if args.epochs is not None:
        cfg.epochs = args.epochs
    out = args.out or RUNS_DIR / args.config / f"seed{args.seed}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_base(GRID["model_name"], device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for checkpointing with a frozen embedding
    apply_lora(model, cfg.target_modules, cfg.rank, cfg.alpha, cfg.dropout, cfg.rslora)

    m = train(model, read_jsonl(DATA_DIR / "train.jsonl"), read_jsonl(DATA_DIR / "val.jsonl"),
              tok.pad_token_id, cfg, out)
    print(f"done: val loss {m['initial_val_loss']:.4f} -> {m['final_val_loss']:.4f}")


if __name__ == "__main__":
    main()
