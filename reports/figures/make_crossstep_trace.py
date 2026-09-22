"""One-off generator: cross-step walkthrough figures for the appendix.

Reads reports/data/xstep_trace.json (measured SmolLM2 trace) and writes:
  xstep_1_pipeline.png   end-to-end flow with real stage counts
  xstep_2_step_detail.png one-step close-up with measured arithmetic
  xstep_3_unrolled.png   positions 0..7 recurrence schematic
  xstep_4_measured.png   measured norms/cosine per position (data-driven)
Rerun: python reports/figures/make_crossstep_trace.py (repo root).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DATA = Path("reports/data/xstep_trace.json")
OUT = Path("reports/figures")
BLUE, RED, GRAY = "#1f77b4", "#c0392b", "#7f8c8d"


def _box(ax, xy, w, h, text, fontsize=8, color="black", face="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                facecolor=face, edgecolor=color))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, wrap=True)


def _arrow(ax, a, b, color="black", w=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=color,
                                 linewidth=w, mutation_scale=11,
                                 shrinkA=2, shrinkB=4))


def fig_pipeline(t):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # NOTE: '$' is stripped ($ opens mathtext in matplotlib labels).
    q = t["question"][:64].replace("$", "")
    stages = [
        (f"GSM8K row\nQ: {q}…\nref ends with '#### 10'", None),
        ("Kojima stage-1 prompt\n'Q: … A: Let's think step by step.'", None),
        (f"Tokenize → {t['n_prefill']} tokens\n(BOS-guarded)", None),
        ("Serial PREFILL, positions 0…44\n"
         "pos 0: warm-up (no mixing)\npos ≥1: mix stored deep→shallow", RED),
        (f"Serial DECODE, {t['n_generated']} tokens\n"
         "sampled greedily, EOS-capped", RED),
        ("Stage-1 reasoning text\n(saved per example)", None),
        ("Stage-2 extraction prompt\n[X′] [reasoning] [Therefore, … is]", None),
        ("Stage-2 forward (recirc active)\n→ final answer text", RED),
        ("Parser: last number\nScorer: == reference ?", None),
    ]
    y, h, gap = 0.93, 0.062, 0.038
    for i, (label, color) in enumerate(stages):
        yy = y - i * (h + gap)
        _box(ax, (0.12, yy - h), 0.76, h, label, fontsize=7.5,
             color=color or "black")
        if i:
            _arrow(ax, (0.5, yy + gap), (0.5, yy))
    ax.set_title("Cross-step GSM8K pass, end to end "
                 f"({t['n_prefill']} prompt + {t['n_generated']} generated tokens)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "xstep_1_pipeline.png", dpi=150)
    plt.close(fig)


def fig_step_detail(t):
    s = t["steps"][0]  # measured position-1 mixing event
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # layer stack (left): 32 blocks, d=4 and s=11 highlighted
    for i in range(32):
        y0 = 0.08 + i * (0.84 / 32)
        if i == 4:
            c, w, lab = BLUE, 2.2, "  destination d=4"
        elif i == 11:
            c, w, lab = RED, 2.2, "  source s=11"
        else:
            c, w, lab = GRAY, 0.7, ""
        ax.add_patch(plt.Rectangle((0.06, y0), 0.16, 0.84 / 32 * 0.92,
                                   facecolor=c if lab else "none",
                                   edgecolor=c, linewidth=w, alpha=0.9
                                   if lab else 0.5))
        if lab:
            ax.text(0.23, y0 + 0.01, lab, fontsize=9, color=c)
    ax.text(0.14, 0.03, "layers 0 (bottom)\nto 31 (top)", ha="center",
            fontsize=7, color=GRAY)
    _arrow(ax, (0.14, 0.08 + 11.5 * 0.84 / 32),
           (0.14, 0.08 + 4.5 * 0.84 / 32), color=RED, w=1.0)
    ax.text(0.02, 0.55, "stored\nfrom\nstep t−1", fontsize=7, color=RED,
            ha="center")
    # measured arithmetic (right)
    txt = (
        f"measured at position 1 (this trace)\n\n"
        f"||s|| = {s['source_norm']:,.1f}\n"
        f"||d|| = {s['destination_norm']:,.1f}   (ratio ≈ "
        f"{s['source_norm'] / s['destination_norm']:.0f}×)\n\n"
        f"f(s) = s × ||d||/||s||   →   ||f(s)|| = "
        f"{s['scaled_source_norm']:,.1f}\n\n"
        f"d′ = 0.15·f(s) + 0.85·d   →   ||d′|| = "
        f"{s['mixed_norm']:,.1f}\n\n"
        f"cos(s, d) = {s['cosine']:.3f}  (shared residual stream)\n\n"
        f"→ d′ feeds block 5; KV for this\n"
        f"   position stores post-mix states"
    )
    _box(ax, (0.42, 0.08), 0.55, 0.84, txt, fontsize=8.5)
    ax.set_title("One cross-step mixing event (measured values)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "xstep_2_step_detail.png", dpi=150)
    plt.close(fig)


def fig_unrolled():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    ax.text(3.5, 2.25, "deep source s=11 (top) → shallow destination d=4 "
            "(bottom), positions 0…7", ha="center", fontsize=10)
    for t in range(8):
        for y, lab in ((1.7, "s"), (0.7, "d")):
            ax.scatter([t], [y], c="black", s=40, zorder=4)
        warm = (t == 0)
        ax.text(t, 0.25, "warm-up\n(no mix)" if warm else f"mix\n(t={t})",
                ha="center", fontsize=7,
                color=RED if not warm else GRAY)
        if t > 0:
            _arc = FancyArrowPatch((t - 1, 1.7), (t, 0.7),
                                   connectionstyle="arc3,rad=0.3",
                                   arrowstyle="-|>", color=RED,
                                   linewidth=1.6, mutation_scale=12)
            ax.add_patch(_arc)
    ax.text(-0.45, 1.7, "deep", fontsize=9, ha="left", va="center")
    ax.text(-0.45, 0.7, "shallow", fontsize=9, ha="left", va="center")
    ax.text(-0.45, 1.2, "KV cache grows\none slot/step", fontsize=7,
            ha="left", va="center", color=GRAY)
    ax.set_title("Unrolled cross-step recurrence (continues through decode)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "xstep_3_unrolled.png", dpi=150)
    plt.close(fig)


def fig_measured(t):
    steps = t["steps"]
    pos = [s["position"] for s in steps]
    src = [s["source_norm"] for s in steps]
    dst = [s["destination_norm"] for s in steps]
    cos = [s["cosine"] for s in steps]
    n_pre = t["n_prefill"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    axes[0].axvspan(0.5, n_pre - 0.5, color="gray", alpha=0.12,
                    label="prefill")
    axes[0].axvspan(n_pre - 0.5, pos[-1] + 0.5, color=BLUE, alpha=0.08,
                    label="decode")
    axes[0].plot(pos, src, color=RED, label="||source|| (deep s=11)")
    axes[0].plot(pos, dst, color=BLUE, label="||destination|| (shallow d=4)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("L2 norm (log scale)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Measured residual norms per position "
                      f"(this trace, {t['model'].split('/')[-1]})", fontsize=11)
    axes[1].axvspan(0.5, n_pre - 0.5, color="gray", alpha=0.12)
    axes[1].axvspan(n_pre - 0.5, pos[-1] + 0.5, color=BLUE, alpha=0.08)
    axes[1].plot(pos, cos, color="black", linewidth=1.2)
    axes[1].set_ylim(0.3, 0.85)
    axes[1].set_xlabel("absolute position (0 = warm-up, no mixing recorded)")
    axes[1].set_ylabel("cos(source, destination)")
    fig.tight_layout()
    fig.savefig(OUT / "xstep_4_measured.png", dpi=150)
    plt.close(fig)


def main():
    t = json.loads((DATA).read_text())
    fig_pipeline(t)
    fig_step_detail(t)
    fig_unrolled()
    fig_measured(t)
    print("wrote xstep_1..4 to", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
