"""Paired baseline-vs-treatment comparison (P2 §19-22).

Usage:
    python scripts/compare_runs.py --baseline <run_A> --treatment <run_B>
    python scripts/compare_runs.py --baseline <run_A> --treatment <run_B> \\
        --output comparisons/my_pair/

Reads the two immutable run dirs (never modifies them) and writes, into a
fresh output dir (default ``comparisons/<A>_vs_<B>/``):

    comparison.json   per-CHANGED-example diff rows (P2 §20 fields)
    comparison.csv    same rows as CSV
    summary.json      cells, deltas, length stats, protocol echoes
    report.txt        human-readable summary

Join key is ``example_id`` (baseline order). ID sets may differ in size
(e.g. --limit 5 vs full); the comparison covers the intersection and says
so loudly. Zero overlap is a hard error.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHANGED_COLUMNS = [
    "example_id",
    "question",
    "reference_answer",
    "baseline_parsed_answer",
    "treatment_parsed_answer",
    "baseline_correct",
    "treatment_correct",
    "baseline_output",
    "treatment_output",
    "baseline_output_tokens",
    "treatment_output_tokens",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    missing = [f for f in ("manifest.json", "metrics.json",
                           "predictions.jsonl") if not (d / f).exists()]
    if not d.is_dir() or missing:
        raise FileNotFoundError(f"run dir {d} missing: {missing or 'not a dir'}")

    def _read_json(name: str) -> dict:
        try:
            return json.loads((d / name).read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"cannot parse {d / name}: {exc}") from exc

    preds = []
    for i, line in enumerate((d / "predictions.jsonl").read_text().splitlines()):
        if not line.strip():
            continue
        try:
            preds.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"cannot parse {d / 'predictions.jsonl'} line {i}: {exc}"
            ) from exc
    return {"dir": str(d), "manifest": _read_json("manifest.json"),
            "metrics": _read_json("metrics.json"), "preds": preds}


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------

def pair_predictions(a_preds: list[dict], b_preds: list[dict]):
    """Join on example_id in baseline order.

    Returns (pairs, a_only, b_only) where pairs are (a, b) tuples.
    """
    by_b = {p["example_id"]: p for p in b_preds}
    pairs, a_only = [], []
    for p in a_preds:
        q = by_b.get(p["example_id"])
        if q is None:
            a_only.append(p["example_id"])
        else:
            pairs.append((p, q))
    b_ids = {p["example_id"] for p in b_preds}
    a_ids = {p["example_id"] for p in a_preds}
    return pairs, a_only, sorted(b_ids - a_ids)


def transition_cells(pairs: list[tuple[dict, dict]]) -> dict[str, int]:
    cells = {"cc": 0, "cw": 0, "wc": 0, "ww": 0}
    for a, b in pairs:
        key = ("c" if a.get("correct") else "w")
        key += ("c" if b.get("correct") else "w")
        cells[key] += 1
    return cells


def changed_rows(pairs: list[tuple[dict, dict]]) -> list[dict]:
    rows = []
    for a, b in pairs:
        if bool(a.get("correct")) == bool(b.get("correct")) and \
                a.get("parsed_answer") == b.get("parsed_answer"):
            continue
        rows.append({
            "example_id": a["example_id"],
            "question": a.get("question"),
            "reference_answer": a.get("reference_answer"),
            "baseline_parsed_answer": a.get("parsed_answer"),
            "treatment_parsed_answer": b.get("parsed_answer"),
            "baseline_correct": bool(a.get("correct")),
            "treatment_correct": bool(b.get("correct")),
            "baseline_output": a.get("raw_output"),
            "treatment_output": b.get("raw_output"),
            "baseline_output_tokens": a.get("output_tokens"),
            "treatment_output_tokens": b.get("output_tokens"),
        })
    return rows


# ---------------------------------------------------------------------------
# Length analysis (P2 §22)
# ---------------------------------------------------------------------------

def _describe(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    s = sorted(values)
    mean = sum(values) / n
    median = float(statistics.median(s))
    p90 = float(s[min(n - 1, int(0.9 * (n - 1)))])
    return {"n": n, "mean": mean, "median": median, "p90": p90}


def length_stats(preds: list[dict]) -> dict:
    """Deterministic length diagnostics per condition.

    Token counts come from the run artifacts when the adapter reported
    them (else null + coverage note); char/word lengths always work.
    ``mean_answer_offset`` is the char offset of the parsed answer's last
    occurrence in the output — a heuristic, documented as such.
    """
    chars = [len(p.get("raw_output") or "") for p in preds]
    words = [len((p.get("raw_output") or "").split()) for p in preds]
    toks = [p.get("output_tokens") for p in preds]
    offsets = []
    for p in preds:
        ans, raw = p.get("parsed_answer"), p.get("raw_output") or ""
        if ans:
            off = raw.rfind(str(ans))
            if off >= 0:
                offsets.append(off)
    tok_known = [t for t in toks if isinstance(t, int)]
    return {
        "n": len(preds),
        "chars": _describe([float(c) for c in chars]),
        "words": _describe([float(w) for w in words]),
        "tokens": _describe([float(t) for t in tok_known]),
        "token_coverage": (len(tok_known) / len(preds)) if preds else 0.0,
        "parse_failures": sum(1 for p in preds if not p.get("parse_success")),
        "mean_answer_offset": (
            sum(offsets) / len(offsets)) if offsets else None,
    }


# ---------------------------------------------------------------------------
# Summary + outputs
# ---------------------------------------------------------------------------

def _cond_block(run: dict, role: str) -> dict:
    m = run["manifest"]
    return {
        "role": role,
        "run_dir": run["dir"],
        "model": m.get("model", {}).get("name"),
        "model_revision": m.get("model", {}).get("revision"),
        "intervention": m.get("model", {}).get("intervention"),
        "experiment": (m.get("experiment", {}) or {}).get("name"),
        "metrics": run["metrics"],
        "task": m.get("task", {}),
        "prompt": m.get("prompt", {}),
        "performance": m.get("performance", {}),
    }


def summarize(baseline: dict, treatment: dict) -> dict:
    pairs, a_only, b_only = pair_predictions(baseline["preds"],
                                            treatment["preds"])
    if not pairs:
        raise ValueError("no common example_ids between the two runs")
    cells = transition_cells(pairs)
    n = len(pairs)
    acc_b, acc_t = (cells["cc"] + cells["cw"]) / n, (cells["cc"] + cells["wc"]) / n
    abs_delta = acc_t - acc_b
    return {
        "baseline": _cond_block(baseline, "baseline"),
        "treatment": _cond_block(treatment, "treatment"),
        "paired": {
            "n_common": n,
            "n_baseline_only": len(a_only),
            "n_treatment_only": len(b_only),
            "cells": cells,
            "baseline_accuracy": acc_b,
            "treatment_accuracy": acc_t,
            "absolute_delta": abs_delta,
            "delta_accuracy": abs_delta,  # P2 §21 descriptive form
            "relative_delta": (abs_delta / acc_b) if acc_b else None,
            "fixed_correct": cells["wc"],
            "rescued_examples": cells["wc"],
            "fixed_incorrect": cells["cw"],
            "regressed_examples": cells["cw"],
        },
        "lengths": {
            "baseline": length_stats([a for a, _ in pairs]),
            "treatment": length_stats([b for _, b in pairs]),
        },
        "n_changed": None,  # filled by caller after changed_rows
    }


def render_report(summary: dict, n_changed: int) -> str:
    p = summary["paired"]
    lines = [
        "Paired comparison",
        f"  baseline : {summary['baseline']['run_dir']}",
        f"  treatment: {summary['treatment']['run_dir']}",
        f"  common examples: {p['n_common']} "
        f"(baseline-only {p['n_baseline_only']}, "
        f"treatment-only {p['n_treatment_only']})",
        "Cells (baseline rows -> treatment cols):",
        f"  correct -> correct: {p['cells']['cc']}",
        f"  correct -> wrong:   {p['cells']['cw']}  (regressed)",
        f"  wrong   -> correct: {p['cells']['wc']}  (rescued)",
        f"  wrong   -> wrong:   {p['cells']['ww']}",
        f"Accuracy: baseline={p['baseline_accuracy']:.4f} "
        f"treatment={p['treatment_accuracy']:.4f} "
        f"delta={p['absolute_delta']:+.4f}",
        f"Changed examples: {n_changed}",
    ]
    for cond in ("baseline", "treatment"):
        ln = summary["lengths"][cond]
        ch = ln["chars"]
        cov = ln["token_coverage"] if "token_coverage" in ln else None
        lines.append(
            f"Lengths [{cond}]: mean_chars={ch['mean']:.1f} "
            f"median={ch['median']:.1f} p90={ch['p90']:.1f} "
            f"parse_failures={ln['parse_failures']}"
            + (f" token_coverage={cov:.2f}" if cov is not None else ""))
    return "\n".join(lines) + "\n"


def write_comparison(baseline_dir: str | Path, treatment_dir: str | Path,
                     output_dir: str | Path | None = None) -> dict:
    treatment_dir = Path(treatment_dir)
    baseline = load_run(baseline_dir)
    treatment = load_run(treatment_dir)
    pairs, _, _ = pair_predictions(baseline["preds"], treatment["preds"])
    rows = changed_rows(pairs)
    summary = summarize(baseline, treatment)
    summary["n_changed"] = len(rows)
    if output_dir is None:
        output_dir = Path("comparisons") / (
            f"{Path(baseline['dir']).name}_vs_{treatment_dir.name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHANGED_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CHANGED_COLUMNS})
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = render_report(summary, len(rows))
    (output_dir / "report.txt").write_text(report)
    return {"output_dir": str(output_dir), "summary": summary,
            "report": report, "n_changed": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Paired baseline-vs-treatment run comparison")
    ap.add_argument("--baseline", required=True, help="Baseline run dir")
    ap.add_argument("--treatment", required=True, help="Treatment run dir")
    ap.add_argument("--output", default=None, help="Output dir")
    ap.add_argument("--wandb-project", default=None,
                    help="Log summary to this W&B project (needs WANDB_API_KEY)")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity (defaults to your account)")
    args = ap.parse_args(argv)
    try:
        result = write_comparison(args.baseline, args.treatment, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"compare_runs: error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    print(f"Artifacts: {result['output_dir']}")
    if args.wandb_project:
        from eval_harness.runtime import wandb_logging
        bname = Path(args.baseline).name
        tname = Path(args.treatment).name
        url = wandb_logging.log_comparison(
            args.wandb_project, result["summary"],
            entity=args.wandb_entity, name=f"compare-{bname}-vs-{tname}")
        print(f"W&B comparison: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
