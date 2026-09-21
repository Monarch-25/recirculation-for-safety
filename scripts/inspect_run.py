"""Inspect a run directory: metrics, manifest summary, paired comparison.

Usage:
    python scripts/inspect_run.py results/gsm8k/.../run_XXX
    python scripts/inspect_run.py results/gsm8k/.../run_A results/.../run_B  # paired
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    preds = [json.loads(l) for l in open(run_dir / "predictions.jsonl")
             if l.strip()]
    metrics = json.load(open(run_dir / "metrics.json"))
    manifest = json.load(open(run_dir / "manifest.json"))
    return {"dir": str(run_dir), "preds": preds,
            "metrics": metrics, "manifest": manifest}


def show(run) -> None:
    m = run["manifest"]
    print(f"Run: {run['dir']}")
    print(f"  model: {m['model']['name']} rev={m['model']['revision']}")
    print(f"  task: {m['task']['name']} split={m['task']['split']}")
    print(f"  metrics: {run['metrics']}")


def paired(a, b) -> None:
    by_id_b = {p["example_id"]: p for p in b["preds"]}
    cells = {"cc": 0, "cw": 0, "wc": 0, "ww": 0}
    for p in a["preds"]:
        q = by_id_b.get(p["example_id"])
        if q is None:
            continue
        key = ("c" if p["correct"] else "w") + ("c" if q["correct"] else "w")
        cells[key] += 1
    print("Paired comparison (A rows, B cols):")
    print(f"  A-correct -> B-correct: {cells['cc']}")
    print(f"  A-correct -> B-wrong:   {cells['cw']}")
    print(f"  A-wrong   -> B-correct: {cells['wc']}")
    print(f"  A-wrong   -> B-wrong:   {cells['ww']}")


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    runs = [load_run(Path(a)) for a in argv[1:]]
    for r in runs:
        show(r)
    if len(runs) == 2:
        paired(runs[0], runs[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
