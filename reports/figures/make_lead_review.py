"""Lead-review figures: architecture, cost, block-uniformity, schedule-indifference.

Reads only frozen committed data (reports/data/*.json, *.csv) and writes:
  figH_architecture.png    harness + evidence-flow diagram
  figH_cost_breakdown.png  metered + reported GPU-hours per experiment
  figH_blocks.png          4B base vs recirc per 100-example block (uniform null)
  figH_schedules.png       rescued/regressed: cross-step twins vs two-pass cells
  figI_transitions_100.png paired 2x2 matrices at n=100, both schedules
    (two-pass cc/ww derived exactly: same-100 baseline is 25/100, so
    cc = 25 - cw and ww = 75 - wc; cross-step rows read verbatim from
    data/sweep_4b_100_transitions.csv)
Rerun: conda run -n torch python reports/figures/make_lead_review.py (repo root).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DATA = Path("reports/data")
OUT = Path("reports/figures")
BASE, RECIRC, GRAY, GREEN = "#8c564b", "#1f77b4", "#7f8c8d", "#2e7d32"
DPI = 150


def _box(ax, xy, w, h, text, fontsize=8, color="black", face="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                facecolor=face, edgecolor=color))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, wrap=True)


def _arrow(ax, a, b, color="black", w=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=color,
                                 linewidth=w, mutation_scale=11,
                                 shrinkA=2, shrinkB=4))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Harness + evidence flow (all arrows regenerate)",
            ha="center", fontsize=11, fontweight="bold")
    # Row 1: define
    _box(ax, (0.02, 0.76), 0.28, 0.12, "GSM8K test, 1319\npinned rev 740312ad")
    _box(ax, (0.36, 0.76), 0.28, 0.12, "Task: 2-stage Kojima\nparser/scorer v1.0")
    _box(ax, (0.70, 0.76), 0.28, 0.12, "Evaluator.evaluate\nseed 42, budgets 256+32")
    # Row 2: execute
    _box(ax, (0.02, 0.55), 0.28, 0.12, "vLLM baseline\nintervention: none", face="#f2f2f2")
    _box(ax, (0.36, 0.55), 0.28, 0.12, "HF recirc adapter\ncross-step / two-pass",
         color=RECIRC, face="#e8f1fa")
    _box(ax, (0.70, 0.55), 0.28, 0.12, "Gemma3 1B/4B PT\npre-paper weights")
    # Row 3: record
    _box(ax, (0.02, 0.34), 0.62, 0.12, "Immutable run dirs: manifest, config,\nmetrics, JSONL, env, logs")
    _box(ax, (0.68, 0.34), 0.30, 0.12, "W&B mirror\nproject recirc-gsm8k", face="#f2f2f2")
    # Row 4: derive
    _box(ax, (0.02, 0.13), 0.28, 0.12, "compare_runs.py\npaired CSVs (frozen)",
         color=GREEN, face="#e9f4ea")
    _box(ax, (0.36, 0.13), 0.28, 0.12, "gsm8k_phase2.py + notebook\nstdlib-only stats",
         color=GREEN, face="#e9f4ea")
    _box(ax, (0.70, 0.13), 0.28, 0.12, "Figures + reports\n(this bundle)",
         color=GREEN, face="#e9f4ea")
    for x in (0.16, 0.50, 0.84):
        _arrow(ax, (x, 0.755), (x, 0.675))
    _arrow(ax, (0.16, 0.545), (0.16, 0.465))
    _arrow(ax, (0.50, 0.545), (0.50, 0.465))
    _arrow(ax, (0.64, 0.40), (0.68, 0.40))
    _arrow(ax, (0.24, 0.34), (0.24, 0.255))
    _arrow(ax, (0.16, 0.125), (0.36, 0.125))
    _arrow(ax, (0.64, 0.19), (0.70, 0.19))
    ax.text(0.5, 0.03, "Green = committed frozen derivatives; every number regenerates from run dirs.",
            ha="center", fontsize=8, color=GRAY)
    fig.savefig(OUT / "figH_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_cost():
    runs = json.load(open(DATA / "runs.json"))
    measured = [
        ("1B pair", runs["pair_1b"]["baseline"]["performance"]["wall_clock_seconds"]
         + runs["pair_1b"]["treatment"]["performance"]["wall_clock_seconds"]),
        ("4B pair", runs["pair_4b"]["baseline"]["performance"]["wall_clock_seconds"]
         + runs["pair_4b"]["treatment"]["performance"]["wall_clock_seconds"]),
    ]
    reported = [("smokes*", 0.5 * 3600), ("sweep + two-pass*", 5.0 * 3600),
                ("heatmap + reruns*", 1.0 * 3600)]
    labels = [k for k, _ in measured] + [k for k, _ in reported]
    hours = [v / 3600 for _, v in measured] + [v / 3600 for _, v in reported]
    colors = [RECIRC, RECIRC] + [GRAY] * 3
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, hours, color=colors)
    for b, h in zip(bars, hours):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.08, f"{h:.2f} h",
                ha="center", fontsize=10)
    ax.set_ylabel("A100-40GB GPU-hours")
    ax.set_title("Phase-2 cost: metered run walls (blue) + W&B-reported estimates (gray *)\n"
                 "total ≈ 8.4 h — Phase-3 ask of 25 h is ~3× all of Phase 2")
    ax.set_ylim(0, 6.2)
    fig.savefig(OUT / "figH_cost_breakdown.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_blocks():
    rows = list(csv.DictReader(open(DATA / "pair_4b_paired.csv")))
    b = [int(r["baseline_correct"]) for r in rows]
    t = [int(r["treatment_correct"]) for r in rows]
    nb = 13
    labels = [f"{i * 100 + 1}-{min((i + 1) * 100, 1319)}" for i in range(nb)]
    base = [sum(b[i * 100:(i + 1) * 100]) / len(b[i * 100:(i + 1) * 100])
            for i in range(nb)]
    rec = [sum(t[i * 100:(i + 1) * 100]) / len(t[i * 100:(i + 1) * 100])
           for i in range(nb)]
    x = range(nb)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([i - 0.2 for i in x], base, width=0.4, color=BASE, label="baseline")
    ax.bar([i + 0.2 for i in x], rec, width=0.4, color=RECIRC, label="recirc")
    ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (n≈100 per block)")
    ax.set_title("4B null is uniform across difficulty spectrum — no hidden subgroup win\n"
                 "block 1 = the sweep subset (base 0.24 vs 0.30 rest)")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "figH_blocks.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_schedules():
    sweep = {r["tag"]: r for r in
             csv.DictReader(open(DATA / "sweep_4b_100.csv"))}
    two = list(csv.DictReader(open(DATA / "twopass_4b_100.csv")))
    cfgs = [("a010_s18_d7", "2p_a010_s18_d7"),
            ("a004_s20_d9", "2p_a004_s20_d9"),
            ("a015_s20_d9", "2p_a015_s20_d9"),
            ("a015_s18_d9", "2p_a015_s18_d9")]
    labels, cr, cw, tr, tw = [], [], [], [], []
    for cs, tp in cfgs:
        labels.append(cs)
        cr.append(int(sweep[cs]["rescued"]))
        cw.append(int(sweep[cs]["regressed"]))
        row = next(r for r in two if r["tag"] == tp)
        tr.append(int(row["rescued_vs_base"]))
        tw.append(int(row["regressed_vs_base"]))
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([i - 0.3 for i in x], cr, width=0.2, color=RECIRC, label="cross-step rescued")
    ax.bar([i - 0.1 for i in x], cw, width=0.2, color=RECIRC, alpha=0.45,
           label="cross-step regressed")
    ax.bar([i + 0.1 for i in x], tr, width=0.2, color=GREEN, label="two-pass rescued")
    ax.bar([i + 0.3 for i in x], tw, width=0.2, color=GREEN, alpha=0.45,
           label="two-pass regressed")
    ax.set_xticks(list(x), labels, fontsize=9)
    ax.set_ylabel("examples (n=100 per cell)")
    ax.set_title("Both schedules move verdicts both ways at similar magnitude\n"
                 "schedule-indifference within noise (±0.07 acc)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "figH_schedules.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figI_transitions_100():
    trans = {r["tag"]: r for r in
             csv.DictReader(open(DATA / "sweep_4b_100_transitions.csv"))}
    two = {r["tag"]: r for r in
           csv.DictReader(open(DATA / "twopass_4b_100.csv"))}
    cfgs = [("a010_s18_d7", "2p_a010_s18_d7"),
            ("a004_s20_d9", "2p_a004_s20_d9"),
            ("a015_s20_d9", "2p_a015_s20_d9"),
            ("a015_s18_d9", "2p_a015_s18_d9")]
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.2))
    for j, (cs, tp) in enumerate(cfgs):
        c = trans[cs]
        mats = {
            "cross-step": [[int(c["ww"]), int(c["wc"])],
                           [int(c["cw"]), int(c["cc"])]],
        }
        r = two[tp]
        cw, wc = int(r["regressed_vs_base"]), int(r["rescued_vs_base"])
        mats["two-pass"] = [[75 - wc, wc], [cw, 25 - cw]]
        for i, (sched, mat) in enumerate(mats.items()):
            ax = axes[i][j]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=75)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, mat[a][b], ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if mat[a][b] > 37 else "black")
            arm = "recirc" if sched == "cross-step" else "two-pass"
            ax.set_xticks([0, 1], [f"{arm}\nwrong", f"{arm}\ncorrect"], fontsize=7)
            if i == 0:
                ax.tick_params(labelbottom=False)
                ax.set_xticks([])
            ax.set_yticks([0, 1], ["base\nwrong", "base\ncorrect"], fontsize=7)
            ax.tick_params(labelright=False)
            acc = (mat[1][1] + mat[0][1]) / 100
            ax.set_title(f"{cs}\n{sched} (acc {acc:.2f})", fontsize=9)
    fig.suptitle("Paired transitions at n=100 — both methods move verdicts both ways\n"
                 "two-pass cc/ww derived exactly from 25/100 baseline + flips",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88], h_pad=3.0)
    fig.savefig(OUT / "figI_transitions_100.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_cost()
    fig_blocks()
    fig_schedules()
    figI_transitions_100()
    print("wrote figH_*.png + figI_transitions_100.png")
