"""Intervention grid sweeps (P2 §41-44, §46-48).

Generates one config per grid cell from a pinned base config, tracks
completed runs, and aggregates per-cell accuracy + paired deltas vs a
baseline run. Diagnostic use: response-surface mapping, NOT official
configuration selection (see plan §43 on test leakage).

Generate:
    python scripts/sweep.py --base configs/gsm8k_gemma3_4b_pt_recirc.yaml \\
        --alphas 0.04,0.07,0.10,0.15 --pairs 18-9,16-9,20-9,18-7 \\
        --limit 100 --out configs/generated

Record each completed run (run dir printed by modal_app as "Done. ..."):
    python scripts/sweep.py --record sweep_4b.json --tag a015_s18_d9 \\
        --config configs/generated/<file>.yaml --run-dir <dir>

Aggregate (needs baseline run dir + all treatment dirs resolvable):
    python scripts/compare_runs.py ... (per cell, or)
    python scripts/sweep.py --aggregate sweep_4b.json \\
        --baseline <baseline_run_dir> --runs-root /tmp/sweep4b \\
        --output comparisons/sweep_4b_100

For --aggregate, each run dir must contain manifest/metrics/predictions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Intervention grid sweeps")
    ap.add_argument("--base", help="Base YAML config")
    ap.add_argument("--alphas", default="",
                    help="Comma list, e.g. 0.04,0.07,0.10,0.15")
    ap.add_argument("--pairs", default="",
                    help="Comma list src-dst, e.g. 18-9,16-9")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="configs/generated")
    ap.add_argument("--schedule", default="cross_step",
                    help="cross_step | two_pass (recorded in manifest)")
    ap.add_argument("--tag-suffix", default="",
                    help="Appended to cell tags, e.g. _2p")
    ap.add_argument("--record", default=None, help="Runs JSON to append to")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--aggregate", default=None,
                    help="Runs JSON to aggregate")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--runs-root", default=None,
                    help="Local dir holding pulled run dirs")
    ap.add_argument("--output", default=None)
    return ap.parse_args(argv)


def _tag(alpha: float, src: int, dst: int, suffix: str = "") -> str:
    return f"a{alpha:04.2f}_s{src}_d{dst}{suffix}".replace(".", "")


def generate(base_path: str, alphas: list[float],
             pairs: list[tuple[int, int]], limit: int,
             out_dir: str, schedule: str = "cross_step",
             tag_suffix: str = "") -> list[dict]:
    import yaml
    base = yaml.safe_load(Path(base_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for alpha in alphas:
        for src, dst in pairs:
            cfg = copy.deepcopy(base)
            cfg["model"]["intervention"].update({
                "source_layer": src, "destination_layer": dst,
                "alpha": alpha, "schedule": schedule,
            })
            tag = _tag(alpha, src, dst, tag_suffix)
            exp = base["experiment"]["name"]
            cfg["experiment"]["name"] = f"{exp}_{tag}"
            cfg["task"]["limit"] = limit
            fname = f"{Path(base_path).stem}_{tag}.yaml"
            (out / fname).write_text(yaml.safe_dump(cfg, sort_keys=False))
            cells.append({"tag": tag, "alpha": alpha,
                          "source_layer": src, "destination_layer": dst,
                          "config": str(out / fname), "run_dir": None})
    return cells


def record(runs_path: str, tag: str, config: str, run_dir: str) -> None:
    p = Path(runs_path)
    data = json.loads(p.read_text()) if p.exists() else {"cells": []}
    # Backfill grid params (alpha/layers) from the generated sweep file so
    # cells recorded out-of-band stay analyzable.
    grid = {}
    grid_path = Path("sweep_runs.json")
    if grid_path.exists():
        for c in json.loads(grid_path.read_text()).get("cells", []):
            grid[c["tag"]] = c
    for c in data["cells"]:
        if c["tag"] == tag:
            merged = {**grid.get(tag, {}), **c}
            merged.update({"config": config, "run_dir": run_dir})
            data["cells"] = [merged if x["tag"] == tag else x
                             for x in data["cells"]]
            break
    else:
        merged = {**grid.get(tag, {}), "tag": tag, "config": config,
                  "run_dir": run_dir}
        data["cells"].append(merged)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.record:
        if not (args.tag and args.config and args.run_dir):
            print("need --tag --config --run-dir with --record",
                  file=sys.stderr)
            return 1
        record(args.record, args.tag, args.config, args.run_dir)
        print(f"recorded {args.tag}")
        return 0
    if args.aggregate:
        return do_aggregate(args)
    if not args.base:
        print("need --base (or --record/--aggregate)", file=sys.stderr)
        return 1
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    pairs = []
    for spec in args.pairs.split(","):
        spec = spec.strip()
        if spec:
            s, d = spec.split("-")
            pairs.append((int(s), int(d)))
    cells = generate(args.base, alphas, pairs, args.limit, args.out,
                     schedule=args.schedule, tag_suffix=args.tag_suffix)
    runs = {"base": args.base, "limit": args.limit, "cells": cells}
    Path("sweep_runs.json").write_text(
        json.dumps(runs, indent=2) + "\n")
    print(f"wrote {len(cells)} configs to {args.out}/ + sweep_runs.json")
    for c in cells:
        print(f"  {c['tag']}: {c['config']}")
    return 0


def do_aggregate(args) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_runs",
        Path(__file__).resolve().parent / "compare_runs.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    runs = json.loads(Path(args.aggregate).read_text())
    baseline = cr.load_run(args.baseline)
    rows = []
    for cell in runs["cells"]:
        rd = ((Path(args.runs_root) / cell["tag"]) if args.runs_root
              else Path(cell["run_dir"]))
        treatment = cr.load_run(str(rd))
        summary = cr.summarize(baseline, treatment)
        p = summary["paired"]
        rows.append({
            "tag": cell["tag"], "alpha": cell.get("alpha"),
            "source_layer": cell.get("source_layer"),
            "destination_layer": cell.get("destination_layer"),
            "accuracy": treatment["metrics"]["accuracy"],
            "delta": p["absolute_delta"],
            "rescued": p["rescued_examples"],
            "regressed": p["regressed_examples"],
            "n": p["n_common"],
        })
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "sweep_table.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    print(f"aggregated {len(rows)} cells -> {out}")
    for r in sorted(rows, key=lambda r: -r["accuracy"]):
        print(f"  {r['tag']}: acc={r['accuracy']:.4f} "
              f"delta={r['delta']:+.4f} (+{r['rescued']}/-{r['regressed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
