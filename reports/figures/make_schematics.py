"""One-off generator: cross-step vs two-pass recurrence schematics.

Our implementation's two schedules, drawn on the same axes conventions
as the paper (input steps horizontal, layer depth vertical). Rerun:
python reports/figures/make_schematics.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle

SRC, DST = 0.78, 0.25  # deep source / shallow destination depths (0..1)
SHALLOW_TOP = 0.42


def _stack(ax, x0, width, label, alpha=1.0):
    ax.add_patch(Rectangle((x0, 0.05), width, 0.9, fill=False,
                           edgecolor="black", linewidth=1.2, alpha=alpha))
    for frac in (0.05 + DST * 0.9, 0.05 + SRC * 0.9):
        ax.plot([x0, x0 + width], [frac, frac], color="gray",
                linewidth=0.8, linestyle=":", alpha=alpha)
    ax.text(x0 + width / 2, 0.0, label, ha="center", va="top", fontsize=8,
            alpha=alpha)


def _dot(ax, x, depth, color, s=70, z=4):
    ax.scatter([x], [0.05 + depth * 0.9], c=color, s=s, zorder=z)


def _arc(ax, x0, d0, x1, d1, color, style="-", w=1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, 0.05 + d0 * 0.9), (x1, 0.05 + d1 * 0.9),
        connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
        color=color, linewidth=w, linestyle=style,
        mutation_scale=12))


def _panel_cross(ax):
    ax.set_title("A. Cross-step (ours, plan §26)", fontsize=11)
    xs = [0.1, 0.4, 0.7]
    for i, x in enumerate(xs):
        _stack(ax, x, 0.18, f"t+{i}" if i else "t (warm-up)")
    # stored source t -> shallow t+1, then t+1 -> t+2
    _dot(ax, xs[0] + 0.09, SRC, "tab:red")
    _dot(ax, xs[1] + 0.09, DST, "tab:blue")
    _arc(ax, xs[0] + 0.09, SRC, xs[1] + 0.09, DST, "tab:red")
    _dot(ax, xs[1] + 0.09, SRC, "tab:red")
    _dot(ax, xs[2] + 0.09, DST, "tab:blue")
    _arc(ax, xs[1] + 0.09, SRC, xs[2] + 0.09, DST, "tab:red")
    ax.text(0.5, 0.97, "one forward per step; signal lags one input",
            ha="center", fontsize=8, style="italic")
    return [("deep source h_s(t), stored", "tab:red"),
            ("mixed destination at t+1", "tab:blue")]


def _panel_twopass(ax):
    ax.set_title("B. Two-pass (paper Fig-3c reading)", fontsize=11)
    xs = [0.08, 0.30, 0.46, 0.68]  # t | t+1 p1, t+1 p2 | t+2 ...
    _stack(ax, xs[0], 0.14, "t (warm-up)", alpha=0.55)
    _stack(ax, xs[1], 0.12, "t+1 pass 1")
    _stack(ax, xs[2], 0.12, "t+1 pass 2")
    _stack(ax, xs[3], 0.14, "t+2 …", alpha=0.55)
    # same-step source -> destination within t+1
    _dot(ax, xs[1] + 0.06, SRC, "tab:red")
    _dot(ax, xs[2] + 0.06, DST, "tab:blue")
    _arc(ax, xs[1] + 0.06, SRC, xs[2] + 0.06, DST, "tab:red")
    ax.plot([xs[2] + 0.06], [0.97], marker=(5, 1), markersize=12,
            color="gold", markeredgecolor="black", zorder=5)
    ax.text(xs[2] + 0.06, 0.80, "logits\nfrom rerun", ha="center",
            fontsize=7)
    ax.text(0.5, 0.97, "two forwards per step; KV overwritten by rerun",
            ha="center", fontsize=8, style="italic")
    return []


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    handles = []
    for ax, fn in zip(axes, (_panel_cross, _panel_twopass)):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 1.0)
        ax.set_xlabel("input steps →")
        ax.set_xticks([])
        ax.set_yticks([0.05 + DST * 0.9, 0.05 + SRC * 0.9],
                      ["shallow\ndestination d", "deep\nsource s"])
        handles += fn(ax)
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="tab:red", marker="o", linestyle="None",
                      label="deep source state"),
        mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="None",
                      label="mixed destination"),
        mlines.Line2D([], [], color="black", marker=(5, 1), linestyle="None",
                      label="readout (two-pass rerun)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8)
    fig.suptitle("Recirculation schedules as implemented "
                 "(d = α·f(s) + β·d, dest-L2 f)", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out = "reports/figures/schematic_recirculation_schedules.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    raise SystemExit(main())
