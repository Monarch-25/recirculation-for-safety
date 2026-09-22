# reports/ — Phase 2 evidence bundle

Self-contained, committable record of the Phase 2 GSM8K experiments.
Everything here regenerates from `data/`; nothing is hand-copied.

## Contents

| path | what |
|---|---|
| `phase2_gsm8k_report.md` | formal report (plan §56, 12 sections) |
| `research_handover_recirculation.md` | **lead handover**: exec summary, methods deep-dive, verdict, Phase-3 proposal + budget |
| `phase2_analysis.ipynb` | executable notebook (validated: every cell runs) |
| `data/pair_{1b,4b}_paired.csv` | frozen per-example derivatives (1319 rows each) |
| `data/twopass_4b_100.csv` | two-pass panel (4 cells vs baseline + vs cross-step) |
| `data/sweep_4b_100.csv` | frozen diagnostic sweep (18 cells: acc, delta, rescued/regressed, McNemar p) |
| `data/sweep_4b_100_transitions.csv` | per-cell paired cc/cw/wc/ww vs same-100 baseline |
| `data/runs.json` | run metadata (models, revisions, metrics, perf) |
| `figures/figA_accuracy.png` | baseline vs recirc + Wilson CIs |
| `figures/figB_transitions.png` | paired transition matrices |
| `figures/figC_lengths.png` | output-length distributions |
| `figures/fig{A,B,C}_{1b,4b}.png` | per-model panels for the sectioned notebook |
| `figures/figD_alpha_sweep.png` | 4B alpha-response curves (diagnostic n=100) |
| `figures/figE_sweep_deltas.png` | per-cell paired deltas vs same-100 baseline |
| `figures/figF_sweep_transitions.png` | rescued-vs-regressed scatter per cell |
| `figures/figG_layer_heatmap.png` | accuracy + delta heatmaps (α=0.10, 3×2) |
| `figures/schematic_recirculation_schedules.png` | cross-step vs two-pass mechanics (generated; build via `figures/make_schematics.py`) |
| `figures/paper_looping_vs_recirc.png` | paper Fig 8, recirc-vs-looping (CC BY-NC-SA 4.0, attributed) |
| `figures/paper_hyperparm_sweep.png` | paper Fig 5, 1B sweep landscape (CC BY-NC-SA 4.0, attributed) |

A full layer heatmap remains future work (the grid is too sparse for one).

## Reproduce

```bash
# figures + summary table from frozen data (no GPU, no downloads):
python analysis/gsm8k_phase2.py --data reports/data --figures reports/figures

# same via the notebook (kernel cwd = reports/):
jupyter nbconvert --to notebook --execute reports/phase2_analysis.ipynb
```

## Provenance

- `data/` derived from the immutable Modal run dirs
  (`run_20260921_213746_f0d8ab`, `run_20260921_213557_173592`,
  `run_20260921_214339_11eb2e`, `run_20260921_213705_b30f77`)
  via exact-join on `example_id` (1319/1319, zero orphans).
- Frozen protocol: `docs/phase2_gsm8k_protocol.md`.
- Changed-question digests (full traces): `comparisons/` (not committed;
  regenerate with `scripts/compare_runs.py`).
- Tracking mirrors: W&B project `recirc-gsm8k` (eval + comparison runs).
