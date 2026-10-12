"""Short runs over each config's LR grid; keeps the LR with the lowest val loss.

Writes runs/selected_lr.json and runs/sweep_table.json.
"""

import json
import subprocess
import sys

from _common import GRID, ROOT, RUNS_DIR


def main():
    sweep = GRID["lr_sweep"]
    chosen, table = {}, {}
    for name, spec in GRID["configs"].items():
        table[name] = {}
        for lr in spec["lr_grid"]:
            out = RUNS_DIR / "sweep" / name / f"lr{lr:g}"
            if not (out / "metrics.json").exists():
                subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", name,
                                "--seed", str(sweep["seed"]), "--lr", str(lr),
                                "--epochs", str(sweep["epochs"]), "--out", str(out)], check=True)
            table[name][lr] = json.loads((out / "metrics.json").read_text())["final_val_loss"]
        chosen[name] = min(table[name], key=table[name].get)
        edges = min(spec["lr_grid"]), max(spec["lr_grid"])
        if chosen[name] in edges:
            print(f"warning: {name} best lr {chosen[name]} is at the edge of its grid; consider widening it")
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "selected_lr.json").write_text(json.dumps(chosen, indent=2))
    (RUNS_DIR / "sweep_table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps({"val_loss": table, "selected": chosen}, indent=2))


if __name__ == "__main__":
    main()
