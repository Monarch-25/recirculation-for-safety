# Recirculation for Safety

> **Research question:** does deep-to-shallow representation recirculation change LLM behavior — reasoning capability first, safety behavior next — without changing model weights?
>
> **Status (2026-09-22):** Phase 1 ✅ · Phase 2 core ✅ · Phase 3 next.
> **Headline result:** no measured GSM8K accuracy effect on Gemma3 1B/4B
> under a frozen, fully traceable protocol (1B: 0.0167→0.0144, p=0.74;
> 4B: 0.2942→0.2775, p=0.26) — with decisive trajectory rewriting
> (69–90% of outputs changed). Independent replication, non-confirming.
> Details: [`reports/research_handover_recirculation.md`](reports/research_handover_recirculation.md).

An inference-time intervention (`d = α·f(s) + β·d`, deep source →
shallow destination) evaluated through a model-agnostic harness where
the *only* moving part between conditions is the inference adapter.
If you are new here, read [`docs/README.md`](docs/README.md) for the
research narrative, then come back for the machinery below.

## Contents

- [Key results](#key-results)
- [Repository tour](#repository-tour)
- [Quickstart (Mac, no GPU)](#quickstart-mac-no-gpu)
- [Reproducing the headline runs (GPU)](#reproducing-the-headline-runs-gpu)
- [Working with this repo](#working-with-this-repo)
- [Docs index](#docs-index)
- [Research discipline](#research-discipline)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Key results

Full GSM8K test (n=1319), greedy, two-stage Kojima prompting, pinned
revisions — every number regenerates from `reports/data/`:

| condition | accuracy | Δ vs base | McNemar p |
|---|---|---|---|
| 1B baseline | 0.0167 (22) | — | — |
| 1B + fixed recirc (11→4, α=0.15) | 0.0144 (19) | −0.0023 | 0.742 |
| 4B baseline | 0.2942 (388) | — | — |
| 4B + fixed recirc (18→9, α=0.15, β=1.0) | 0.2775 (366) | −0.0167 | 0.260 |

Plus: 4B diagnostic sweep (18 cells, n=100) with response surface and
heatmaps, and a two-pass schedule panel — see
[`reports/phase2_analysis.ipynb`](reports/phase2_analysis.ipynb).
Changed-question digests (every flipped example with full traces):
`comparisons/` (regenerable, not committed).

## Repository tour

```text
recirculation-for-safety/
├── configs/                  # frozen experiment YAMLs (pinned *_pinned, Gemma, sweep cells generated on demand)
├── scripts/                  # evaluate.py, compare_runs.py, sweep.py, inspect_run.py
├── src/eval_harness/         # core/ evaluator+config+results · models/ hf, vllm, mock, recirculation
│                             # tasks/ · prompting/ · parsing/ · scoring/ · runtime/ · utils/
├── analysis/gsm8k_phase2.py  # stats (Wilson/McNemar, stdlib) + all figures, no hardcoded numbers
├── tests/                    # offline unit + mock integration + model-backed (gated by markers)
├── reports/                  # handover report, executable notebook, frozen data, figures
├── docs/                     # research narrative, plans, frozen protocols, progress trackers
├── comparisons/              # paired-run bundles (local only, regenerable — not committed)
├── results/                  # immutable run dirs (local only — not committed)
├── modal_app.py              # Modal A100 entrypoint (vLLM + HF paths)
└── pyproject.toml            # install with .[dev], .[tracking], .[vllm] extras
```

Design rule, enforced by construction: `Task` owns benchmark semantics,
`ModelAdapter` owns generation, `Evaluator.evaluate(task, model, config)`
orchestrates. Recirculation lives entirely in
`src/eval_harness/models/recirculation.py` (cross-step *and* two-pass
schedules behind one `schedule` flag) — evaluator, tasks, prompts,
parsers, scorers never change between conditions.

## Quickstart (Mac, no GPU)

```bash
conda activate torch
pip install -e ".[dev]"        # offline-capable install

pytest -q                      # full offline suite (no downloads)
pytest -q -m model             # model-backed invariants (uses HF cache when present)

# Dry run: resolves config/env without generating
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --dry-run

# 5-example smoke test (Apple Silicon OK)
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --limit 5

# Paired comparison of any two run dirs (no rerun needed)
python scripts/compare_runs.py --baseline <run_A> --treatment <run_B> --output comparisons/my_pair
```

CLI overrides: `--limit N`, `--batch-size N`, `--device ...`.
Backends: `hf` (CPU/MPS/CUDA) · `vllm` (Linux/CUDA only).

## Reproducing the headline runs (GPU)

Requires Modal access + `huggingface` secret (`HF_TOKEN`, gated Gemma
license) — see `modal_app.py` header. W&B logging is automatic when
`experiment.wandb_project` is set (`wandb` secret provides the key).

```bash
# 1B / 4B baselines (vLLM, fast)
modal run modal_app.py --config configs/gsm8k_gemma3_1b_pt_baseline_vllm.yaml
modal run modal_app.py --config configs/gsm8k_gemma3_4b_pt_baseline_vllm.yaml

# Recirculation treatments (HF serial adapter; ~4s/ex 1B, ~5s/ex 4B)
modal run modal_app.py --config configs/gsm8k_gemma3_1b_pt_recirc.yaml
modal run modal_app.py --config configs/gsm8k_gemma3_4b_pt_recirc.yaml

# 100-sample diagnostic sweep (4B, 18 cells) + two-pass panel
python scripts/sweep.py --base configs/gsm8k_gemma3_4b_pt_recirc.yaml \
  --alphas 0.04,0.07,0.10,0.15 --pairs 18-9,16-9,20-9,18-7 \
  --limit 100 --out configs/generated
# ... run each generated config, record with --record, aggregate, compare

# Regenerate every figure/table from frozen data (no GPU, no downloads)
python analysis/gsm8k_phase2.py --data reports/data --figures reports/figures
```

Sync run artifacts back with
`modal volume get recirc-results <run-path> <local-path>`.
Every run dir carries `manifest.json` (all pins/hashes),
`predictions.jsonl` (per-example), `metrics.json`, `environment.json`,
`logs.txt` — plus a W&B mirror in project `recirc-gsm8k`.

## Working with this repo

- **Add a new condition** (layers/alpha/schedule): edit or generate a YAML —
  `InterventionConfig` validates it loudly (`source>dest`, beta modes,
  known schedules). Never hardcode experiment constants.
- **Add a backend**: implement `ModelAdapter.generate(prompts, config)`.
  Optionally publish `last_batch_info` (timing/token counts); the
  evaluator records nulls for anything unmeasured.
- **Add a benchmark** (Phase 3 safety tasks): new `Task` + parser/scorer.
  The evaluator, adapters, and result schema stay untouched.
- **Analyze**: `analysis/gsm8k_phase2.py` functions + the notebook.
  Figures/tables must come from `reports/data/` — never hand-copied.

## Docs index

| doc | what |
|---|---|
| `docs/README.md` | research narrative (questions, hypotheses, roadmap) |
| `docs/RESEARCH_PROGRESS.md` | **living gated checklist** — start here to see what's done |
| `docs/research_plan_phase1.md` | Phase-1 harness spec |
| `docs/research_plan_phase2.md` | Phase-2 GSM8K/recirculation spec |
| `docs/research_plan_phase3.md` | Phase-3 safety spec (next) |
| `docs/evaluation_protocol.md` | frozen benchmark contract |
| `docs/phase2_gsm8k_protocol.md` | frozen Phase-2 values (do not modify) |
| `docs/progress.md` | session-by-session engineering log |
| `reports/research_handover_recirculation.md` | **lead handover**: verdict, methods, budget |
| `reports/phase2_gsm8k_report.md` | formal 12-section Phase-2 record |
| `reports/phase2_analysis.ipynb` | executable analysis (validated) |

## Research discipline

- Gated phases: code existing ≠ phase complete (`docs/RESEARCH_PROGRESS.md`).
- No test-set tuning; development/final-test separation; frozen configs.
- Paired, example-level analysis — never aggregate-only claims.
- Nulls and failures are research records, not embarrassments (see the
  verdict above and §7/§11 of the handover).
- denoms with every %; CIs and paired tests, not point estimates.

## Roadmap

- [x] **Phase 1** — reproducible evaluation harness.
- [x] **Phase 2 core** — trusted GSM8K baselines, fixed recirculation
      (two schedules), full-set comparisons, sweep + heatmaps, handover.
      Open threads: lm-harness cross-check, pass@128, 12B (all deferred
      by plan, none blocking).
- [ ] **Phase 3** — safety benchmarks (harmful + benign), judges,
      safety–utility matrix. See `docs/research_plan_phase3.md` and the
      handover §9 for the proposal + compute budget.
- [ ] **Phase 4+** — mechanism, ALIGNBEAM comparison, robustness, release.

## Contributing

Research PRs should answer: *what hypothesis/infra need does this
address? what experiment does it enable? what invariant/test guards it?
what metadata gets recorded?* Small, reviewable commits; never commit
secrets (`access_tokens.txt`), run outputs (`results/`, `comparisons/`,
`wandb/`), or generated sweep configs — manifests already record
everything needed to reproduce.

## Citation

If you use this repository, cite the underlying work and record the
commit + config used:

- Mozer, Siddiqui, Sawyer, Sanyal, Liu. *Recirculation.*
  https://arxiv.org/abs/2608.17981 (paper figures reused under CC BY-NC-SA 4.0)
- Kojima et al. *Large Language Models are Zero-Shot Reasoners.* NeurIPS 2022.
- Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. (GSM8K)
- This repo: https://github.com/Monarch-25/recirculation-for-safety
  (+ commit sha + `manifest.json` from your run)

## License

TBD — no license file yet; default is all-rights-reserved until the
maintainer adds one. If you need reuse terms, open an issue.
