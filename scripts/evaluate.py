"""Evaluate the base model or one fine-tuned run on HumanEval and MBPP.

    python scripts/evaluate.py --base
    python scripts/evaluate.py --run runs/lora_r16/seed0

Writes results/<name>/<benchmark>.jsonl (one row per problem, including the
raw response, the executed program's status and stderr tail) and summary.json.
"""

import argparse
import json
from pathlib import Path

import torch
from _common import GRID, RESULTS_DIR, write_jsonl

from lora_code.benchmarks import BENCHMARKS, load_benchmark
from lora_code.execution import run_many
from lora_code.modeling import generate, load_base, load_finetuned
from lora_code.stats import summarize


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", action="store_true")
    g.add_argument("--run")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.base:
        model, tok = load_base(GRID["model_name"], device)
        name = "base"
    else:
        model, tok = load_finetuned(GRID["model_name"], args.run, device)
        run = Path(args.run)
        name = f"{run.parent.name}/{run.name}"

    out_dir = RESULTS_DIR / name
    summary = {}
    for bench in args.benchmarks:
        spec = BENCHMARKS[bench]
        problems = load_benchmark(bench)
        responses = generate(model, tok, [spec["instruction"](p) for p in problems],
                             args.max_new_tokens, args.batch_size)
        programs = [spec["program"](p, r) for p, r in zip(problems, responses)]
        results = run_many(programs, timeout=args.timeout)
        rows = [{"task_id": p[spec["id_key"]], "passed": int(res.passed), "status": res.status,
                 "response": r, "stderr_tail": res.stderr_tail}
                for p, r, res in zip(problems, responses, results)]
        write_jsonl(out_dir / f"{bench}.jsonl", rows)
        summary[bench] = summarize([r["passed"] for r in rows])
        print(bench, json.dumps(summary[bench]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
