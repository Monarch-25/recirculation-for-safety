"""Phase-2 GSM8K analysis (P2 §54): tables + figures from frozen artifacts.

Reads ONLY ``reports/data/{pair_1b,pair_4b}_paired.csv`` + ``runs.json``
(derivatives of the immutable run dirs; see reports/README). Generates
summary statistics and figures; never hardcodes result numbers.

Figures:
    A  baseline vs recirculation accuracy (Wilson 95% CI)
    B  baseline -> recirculation transition matrices
    C  output-length distributions per condition
    (D alpha sweep and E layer heatmap are not generated: no sweep was
    executed in Phase 2. The report states this explicitly.)

Usage:
    python analysis/gsm8k_phase2.py [--data reports/data]
        [--figures reports/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASELINE_COLOR = "#8c564b"  # brown, matching the paper's baseline bars
RECIRC_COLOR = "#1f77b4"  # blue, matching the paper's recirc bars

PAIRS = ("pair_1b", "pair_4b")
PAIR_LABELS = {"pair_1b": "Gemma3 1B PT", "pair_4b": "Gemma3 4B PT"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pair(data_dir: str | Path, tag: str):
    """Return (rows, meta) for one pair.

    rows: list of dicts with int/float fields parsed; meta: runs.json
    slice for the tag.
    """
    data_dir = Path(data_dir)
    with open(data_dir / f"{tag}_paired.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("baseline_correct", "treatment_correct",
                  "baseline_chars", "treatment_chars"):
            r[k] = int(r[k])
        for k in ("baseline_tokens", "treatment_tokens"):
            r[k] = int(r[k]) if r[k] not in ("", None) else None
    meta = json.loads((data_dir / "runs.json").read_text())[tag]
    return rows, meta


# ---------------------------------------------------------------------------
# Statistics (stdlib only; P2 §49)
# ---------------------------------------------------------------------------

def transition_cells(rows) -> dict[str, int]:
    cells = {"cc": 0, "cw": 0, "wc": 0, "ww": 0}
    for r in rows:
        key = ("c" if r["baseline_correct"] else "w")
        key += ("c" if r["treatment_correct"] else "w")
        cells[key] += 1
    return cells


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(cw: int, wc: int) -> tuple[float, float]:
    """McNemar chi2 (continuity-corrected) + two-sided p for paired binary.

    cw = baseline-correct/treatment-wrong, wc = the reverse.
    """
    if cw + wc == 0:
        return (0.0, 1.0)
    chi2 = (abs(cw - wc) - 1) ** 2 / (cw + wc)
    p = math.erfc(math.sqrt(chi2 / 2))
    return (chi2, p)


def pair_stats(rows) -> dict:
    n = len(rows)
    cells = transition_cells(rows)
    nb = cells["cc"] + cells["cw"]
    nt = cells["cc"] + cells["wc"]
    chi2, p = mcnemar(cells["cw"], cells["wc"])
    return {
        "n": n,
        "cells": cells,
        "baseline_accuracy": nb / n,
        "treatment_accuracy": nt / n,
        "absolute_delta": (nt - nb) / n,
        "relative_delta": ((nt - nb) / nb) if nb else None,
        "rescued": cells["wc"],
        "regressed": cells["cw"],
        "baseline_ci": wilson(nb, n),
        "treatment_ci": wilson(nt, n),
        "mcnemar_chi2": chi2,
        "mcnemar_p": p,
        "n_changed": sum(1 for r in rows
                         if r["baseline_correct"] != r["treatment_correct"]
                         or r["baseline_parsed"] != r["treatment_parsed"]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_a(data_dir, out_path: str | Path,
             only: str | None = None) -> Path:
    """Grouped accuracy bars with Wilson 95% CI whiskers.

    ``only="pair_1b"`` (or ``"pair_4b"``) renders a single-model panel
    for sectioned notebooks; default renders both.
    """
    tags = [only] if only else ["pair_1b", "pair_4b"]
    labels, base, rec, blo, bhi, rlo, rhi = [], [], [], [], [], [], []
    for tag in tags:
        rows, _ = load_pair(data_dir, tag)
        s = pair_stats(rows)
        n = s["n"]
        labels.append(PAIR_LABELS[tag])
        base.append(s["baseline_accuracy"])
        rec.append(s["treatment_accuracy"])
        lb, ub = s["baseline_ci"]
        lt, ut = s["treatment_ci"]
        blo.append(s["baseline_accuracy"] - lb)
        bhi.append(ub - s["baseline_accuracy"])
        rlo.append(s["treatment_accuracy"] - lt)
        rhi.append(ut - s["treatment_accuracy"])
    x = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w / 2 for i in x], base, w, label="baseline",
           color=BASELINE_COLOR, yerr=[blo, bhi], capsize=4)
    ax.bar([i + w / 2 for i in x], rec, w, label="recirculation",
           color=RECIRC_COLOR, yerr=[rlo, rhi], capsize=4)
    for i, (b, r) in enumerate(zip(base, rec)):
        ax.text(i, max(b, r) + 0.012, f"Δ {r - b:+.4f}",
                ha="center", fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("GSM8K accuracy (greedy pass@1)")
    title = "Figure A — baseline vs fixed recirculation (n=1319 each)"
    if only:
        title += f" [{PAIR_LABELS[only]}]"
    ax.set_title(title, pad=20 if only else 6)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_b(data_dir, out_path: str | Path,
             only: str | None = None) -> Path:
    """Transition matrices baseline -> recirculation (one panel per model)."""
    tags = ([only] if only else ["pair_1b", "pair_4b"])
    ncols = len(tags)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 3.8))
    if ncols == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        rows, _ = load_pair(data_dir, tag)
        c = transition_cells(rows)
        mat = [[c["ww"], c["wc"]], [c["cw"], c["cc"]]]
        ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], ["recirc wrong", "recirc correct"])
        ax.set_yticks([0, 1], ["base wrong", "base correct"])
        peak = max(max(row) for row in mat)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                        fontsize=12,
                        color="white" if mat[i][j] > peak / 2 else "black")
        ax.set_title(f"{PAIR_LABELS[tag]} (n={len(rows)})")
    title = "Figure B — paired transitions"
    if only:
        title += f" [{PAIR_LABELS[only]}]"
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_c(data_dir, out_path: str | Path,
             only: str | None = None) -> Path:
    """Output-length (chars) distributions per condition."""
    tags = ([only] if only else ["pair_1b", "pair_4b"])
    ncols = len(tags)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.8),
                             sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        rows, _ = load_pair(data_dir, tag)
        bl = [r["baseline_chars"] for r in rows]
        tl = [r["treatment_chars"] for r in rows]
        bins = max(10, min(60, max(bl + tl) // 40 or 10))
        ax.hist(bl, bins=bins, alpha=0.6, label="baseline",
                color=BASELINE_COLOR)
        ax.hist(tl, bins=bins, alpha=0.6, label="recirculation",
                color=RECIRC_COLOR)
        ax.set_xlabel("stage-2 output chars")
        ax.set_title(PAIR_LABELS[tag])
    axes[0].set_ylabel("examples")
    axes[0].legend()
    title = "Figure C — output-length distributions"
    if only:
        title += f" [{PAIR_LABELS[only]}]"
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_e(data_dir, out_path: str | Path) -> Path:
    """Per-cell paired deltas vs the same-100 baseline (diagnostic sweep).

    Horizontal bars sorted by delta; star = nominal McNemar p < 0.05
    (all Bonferroni-n.s. across 16 cells — stated in the title note, not
    hidden). This is the cell-by-cell baseline comparison; Figure D
    shows the same data as response curves.
    """
    rows = sorted(load_sweep(data_dir), key=lambda r: r["delta"])
    tags = [r["tag"] for r in rows]
    deltas = [r["delta"] for r in rows]
    sig = [r["mcnemar_p"] < 0.05 for r in rows]
    colors = [RECIRC_COLOR if d >= 0 else BASELINE_COLOR for d in deltas]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = range(len(tags))
    bars = ax.barh(list(y), deltas, color=colors)
    for i, (d, s) in enumerate(zip(deltas, sig)):
        if s:
            ax.text(d + (0.004 if d >= 0 else -0.004), i, "*",
                    ha="left" if d >= 0 else "right", va="center",
                    fontsize=12, fontweight="bold")
    ax.set_yticks(list(y), tags, fontsize=8)
    ax.set_xlabel("paired accuracy delta vs same-100 baseline (0.25)")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Figure E — sweep cells vs 100-sample baseline "
                 "(* nominal p<0.05; all Bonferroni-n.s.)")
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def load_sweep_transitions(data_dir: str | Path):
    """Per-cell paired transition counts vs the same-100 baseline."""
    with open(Path(data_dir) / "sweep_4b_100_transitions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cc", "cw", "wc", "ww"):
            r[k] = int(r[k])
    return rows


def figure_f(data_dir, out_path: str | Path) -> Path:
    """Rescued vs regressed per sweep cell (diagonal = null effect).

    Points above the diagonal are net-positive cells. Marker size
    encodes nothing — labels carry the tag. The paper configuration
    cell is highlighted.
    """
    rows = load_sweep_transitions(data_dir)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    # Hand-placed offsets for the tight middle cluster; cycled fallback
    # for the rest. Deterministic; data positions untouched.
    manual = {
        "a010_s18_d9": (-40, 8),
        "a007_s20_d9": (-14, -16),
        "a010_s20_d9": (8, -14),
        "a015_s18_d9": (8, 8),
        "a007_s16_d9": (8, 6),
        "a010_s16_d9": (8, -14),
    }
    fallback = [(5, 5), (5, -11), (-32, 5), (-32, -11)]
    for k, r in enumerate(rows):
        is_paper = (r["tag"] == "a015_s18_d9")
        ax.scatter(r["cw"], r["wc"],
                   s=120 if is_paper else 60,
                   color=RECIRC_COLOR if is_paper else "gray",
                   alpha=0.9, zorder=3)
        dx, dy = manual.get(r["tag"], fallback[k % len(fallback)])
        ax.annotate(r["tag"], (r["cw"], r["wc"]),
                    fontsize=7, xytext=(dx, dy),
                    textcoords="offset points")
    lims = [0, max(max(r["cw"], r["wc"]) for r in rows) + 2]
    ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--",
            label="null (rescued = regressed)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("regressed (baseline correct -> recirc wrong)")
    ax.set_ylabel("rescued (baseline wrong -> recirc correct)")
    ax.set_title("Figure F — sweep transitions vs 100-sample baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_g(data_dir, out_path: str | Path,
             alpha: float = 0.10) -> Path:
    """Layer heatmaps at one alpha: accuracy + paired delta.

    Local window around the paper pair (sources {16,18,20} ×
    destinations {7,9}). Fails loudly if the slice is incomplete —
    a heatmap with holes would mislead.
    """
    rows = [r for r in load_sweep(data_dir) if r["alpha"] == alpha]
    srcs = sorted({r["source_layer"] for r in rows})
    dsts = sorted({r["destination_layer"] for r in rows})
    grid = {(r["source_layer"], r["destination_layer"]): r for r in rows}
    missing = [(s, d) for s in srcs for d in dsts if (s, d) not in grid]
    if missing:
        raise ValueError(f"heatmap slice alpha={alpha} missing cells "
                         f"(run them before plotting): {missing}")
    acc = [[grid[(s, d)]["accuracy"] for d in dsts] for s in srcs]
    delta = [[grid[(s, d)]["delta"] for d in dsts] for s in srcs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, mat, title, fmt, cmap, vcenter in (
        (axes[0], acc, "accuracy", "{:.2f}", "Blues", None),
        (axes[1], delta, "paired delta vs baseline",
         "{:+.2f}", "RdBu", 0.0),
    ):
        if vcenter is not None:
            bound = max(abs(v) for row in mat for v in row) or 1.0
            im = ax.imshow(mat, cmap=cmap, vmin=-bound, vmax=bound)
        else:
            im = ax.imshow(mat, cmap=cmap)
        ax.set_xticks(range(len(dsts)), [f"d{d}" for d in dsts])
        ax.set_yticks(range(len(srcs)), [f"s{s}" for s in srcs])
        peak = max(max(row) for row in mat)
        floor = min(min(row) for row in mat)
        for i in range(len(srcs)):
            for j in range(len(dsts)):
                v = mat[i][j]
                frac = ((v - floor) / (peak - floor)) if peak > floor else 0.5
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=11,
                        color="white" if (frac > 0.6 if vcenter is None
                                          else abs(v) > bound * 0.55)
                        else "black")
        ax.set_title(title)
        ax.set_xlabel("destination layer")
        if ax is axes[0]:
            ax.set_ylabel("source layer")
    fig.suptitle(f"Figure G — layer heatmap at alpha={alpha} "
                 f"(diagnostic n=100)")
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def load_sweep(data_dir: str | Path):
    """Load the frozen 4B diagnostic sweep (18 cells, n=100 each)."""
    with open(Path(data_dir) / "sweep_4b_100.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["source_layer"] = int(r["source_layer"])
        r["destination_layer"] = int(r["destination_layer"])
        for k in ("accuracy", "delta", "mcnemar_p", "baseline_accuracy"):
            r[k] = float(r[k])
        for k in ("rescued", "regressed", "n", "baseline_n"):
            r[k] = int(r[k])
    return rows


def figure_d(data_dir, out_path: str | Path,
             baseline_accuracy: float = 0.25) -> Path:
    """Alpha-response curves per layer pair (diagnostic n=100 sweep)."""
    rows = load_sweep(data_dir)
    pairs = sorted({(r["source_layer"], r["destination_layer"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for src, dst in pairs:
        pts = sorted(
            [r for r in rows
             if (r["source_layer"], r["destination_layer"]) == (src, dst)],
            key=lambda r: r["alpha"])
        is_paper = (src, dst) == (18, 9)
        ax.plot([p["alpha"] for p in pts], [p["accuracy"] for p in pts],
                marker="o" if is_paper else "s",
                linewidth=2.2 if is_paper else 1.4,
                label=f"s{src}->d{dst}" + (" (paper)" if is_paper else ""))
    ax.axhline(baseline_accuracy, color="gray", linestyle="--",
               label="baseline (same 100)")
    ax.set_xlabel("alpha (beta=1.0 nonconvex, dest-L2, no ramp)")
    ax.set_ylabel("GSM8K accuracy, first-100 subset")
    ax.set_title("Figure D — 4B alpha response by layer pair (diagnostic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_table(data_dir) -> str:
    lines = ["| model | base acc | recirc acc | Δ | rescued | regressed | "
             "McNemar p |",
             "|---|---|---|---|---|---|---|"]
    for tag, label in (("pair_1b", "1B"), ("pair_4b", "4B")):
        rows, _ = load_pair(data_dir, tag)
        s = pair_stats(rows)
        lines.append(
            f"| {label} | {s['baseline_accuracy']:.4f} | "
            f"{s['treatment_accuracy']:.4f} | {s['absolute_delta']:+.4f} | "
            f"{s['rescued']} | {s['regressed']} | {s['mcnemar_p']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase-2 GSM8K analysis")
    ap.add_argument("--data", default="reports/data")
    ap.add_argument("--figures", default="reports/figures")
    args = ap.parse_args(argv)
    figs = Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    print(summary_table(args.data))
    print("A:", figure_a(args.data, figs / "figA_accuracy.png"))
    print("B:", figure_b(args.data, figs / "figB_transitions.png"))
    print("C:", figure_c(args.data, figs / "figC_lengths.png"))
    sweep_csv = Path(args.data) / "sweep_4b_100.csv"
    if sweep_csv.exists():
        print("D:", figure_d(args.data, figs / "figD_alpha_sweep.png"))
    else:
        print("D: skipped (no sweep_4b_100.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
