"""Same pipeline as run_all.sh, spread over several GPUs.

    python scripts/run_parallel.py --gpus 0,1,2,3

Every sweep run, training run and eval is its own job. A job starts when a GPU
is free and the jobs it depends on have finished. Logs go to logs/.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from _common import GRID, RESULTS_DIR, ROOT, RUNS_DIR

PY = sys.executable
LOGS = ROOT / "logs"


def sweep_dir(name, lr):
    return RUNS_DIR / "sweep" / name / f"lr{lr:g}"


def selected_lr(name):
    """Lowest final validation loss over the grid, as in sweep_lr.py."""
    table = {lr: json.loads((sweep_dir(name, lr) / "metrics.json").read_text())["final_val_loss"]
             for lr in GRID["configs"][name]["lr_grid"]}
    return min(table, key=table.get)


def run_jobs(jobs, gpus):
    """jobs: list of (name, deps, make_cmd) in priority order. A job starts on a free
    GPU once all its deps have finished; make_cmd is called only then."""
    names = {name for name, _, _ in jobs}
    pending = [(name, deps & names, make_cmd) for name, deps, make_cmd in jobs]
    free, running, done, failed = list(gpus), {}, set(), set()
    while pending or running:
        for job in list(pending):
            name, deps, make_cmd = job
            if deps & failed:
                pending.remove(job)
                failed.add(name)
                print(f"SKIPPED {name} (a dependency failed)", flush=True)
            elif free and deps <= done:
                pending.remove(job)
                g, cmd = free.pop(0), make_cmd()
                log = open(LOGS / f"{name}.log", "a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "PYTHONUNBUFFERED": "1"}
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[name] = (proc, g, log, time.time())
                print(f"start {name} on gpu {g}: {' '.join(cmd[1:])}", flush=True)
        time.sleep(5)
        for name, (proc, g, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            free.append(g)
            del running[name]
            if proc.returncode:
                failed.add(name)
                print(f"FAILED {name} (see logs/{name}.log)", flush=True)
            else:
                done.add(name)
                print(f"done {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        raise SystemExit(f"failed: {sorted(failed)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices, e.g. 0,1,2,3")
    gpus = ap.parse_args().gpus.split(",")
    LOGS.mkdir(exist_ok=True)

    subprocess.run([PY, "scripts/prepare_data.py"], cwd=ROOT, check=True)

    sweep, configs = GRID["lr_sweep"], GRID["configs"]
    sweeps, trains, evals = [], [], []
    for name, spec in configs.items():
        for lr in spec["lr_grid"]:
            out = sweep_dir(name, lr)
            if not (out / "metrics.json").exists():
                cmd = [PY, "scripts/train.py", "--config", name, "--seed", str(sweep["seed"]),
                       "--lr", str(lr), "--epochs", str(sweep["epochs"]), "--out", str(out)]
                sweeps.append((f"sweep_{name}_lr{lr:g}", set(), lambda cmd=cmd: cmd))
    if not (RESULTS_DIR / "base" / "summary.json").exists():
        evals.append(("eval_base", set(), lambda: [PY, "scripts/evaluate.py", "--base"]))

    # configs still waiting on a sweep run go first, they're on the longest chain
    unswept = {n for n, *_ in sweeps}
    order = sorted(configs, key=lambda n: not any(s.startswith(f"sweep_{n}_") for s in unswept))
    for name in order:
        deps = {s for s in unswept if s.startswith(f"sweep_{name}_")}
        for seed in GRID["seeds"]:
            run = RUNS_DIR / name / f"seed{seed}"
            if not (run / "adapter.pt").exists():
                trains.append((f"train_{name}_seed{seed}", deps,
                               lambda name=name, seed=seed: [PY, "scripts/train.py", "--config", name,
                                                             "--seed", str(seed), "--lr", str(selected_lr(name))]))
            if not (RESULTS_DIR / name / f"seed{seed}" / "summary.json").exists():
                evals.append((f"eval_{name}_seed{seed}", {f"train_{name}_seed{seed}"},
                              lambda run=run: [PY, "scripts/evaluate.py", "--run", str(run)]))

    run_jobs(sweeps + trains + evals, gpus)
    subprocess.run([PY, "scripts/sweep_lr.py"], cwd=ROOT, check=True)  # just writes selected_lr.json now
    subprocess.run([PY, "scripts/aggregate.py"], cwd=ROOT, check=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
