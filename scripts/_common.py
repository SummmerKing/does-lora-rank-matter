import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GRID = json.loads((ROOT / "configs" / "grid.json").read_text())
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
