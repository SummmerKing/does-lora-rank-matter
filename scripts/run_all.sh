#!/usr/bin/env bash
# Full study. Each step skips work that is already done, so it can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py
python scripts/sweep_lr.py
[ -f results/base/summary.json ] || python scripts/evaluate.py --base

for cfg in lora_r4 lora_r16 lora_r64 loraplus_r16; do
  for seed in 0 1 2; do
    run="runs/${cfg}/seed${seed}"
    [ -f "${run}/adapter.pt" ] || python scripts/train.py --config "$cfg" --seed "$seed"
    [ -f "results/${cfg}/seed${seed}/summary.json" ] || python scripts/evaluate.py --run "$run"
  done
done

python scripts/aggregate.py
