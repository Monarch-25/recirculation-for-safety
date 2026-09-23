# Recirculation for Safety

> **Research question:** does deep-to-shallow representation recirculation change LLM behavior — reasoning capability first, safety behavior next — without changing model weights?
>
> **Status (2026-09-23):** Phase 1 ✅ · Phase 2 core ✅ · Mozer replication ✅ (see below) · Phase 3 next.
> **Headline result:** paper-exact Mozer recirculation reproduces the
> reference regime on GSM8K-Platinum/Gemma-3-1B-IT (ours 531/1209 = 43.92%
> vs repo dense 540/1209 and repo recirc 554/1209; McNemar p=0.58/0.12 —
> indistinguishable). A same-harness delayed-feedback ("cross-step")
> variant scores 559/1209 = 46.24% (+28 paired over mozer, p=0.054 —
> borderline, not significant, and not the paper's method).
> Full paper: [`reports/paper_gsm8k_mt_replication.md`](reports/paper_gsm8k_mt_replication.md).

An inference-time intervention (`d = α·f(s) + β·d`, deep source →
shallow destination) evaluated through a model-agnostic harness where
the *only* moving part between conditions is the inference adapter.
If you are new here, read [`docs/README.md`](docs/README.md) for the
research narrative, then come back for the machinery below.

## Live presentation

- 🌐 **Interactive site:** https://monarch-25.github.io/recirculation-for-safety/ — research notebook microsite (source: `site/`, deployed via `gh-pages`; re-sync after edits with `git subtree push --prefix site origin gh-pages`)
- 🎬 **Slides:** https://monarch-25.github.io/recirculation-for-safety/slides/recirculation.html — 26-slide methodology deck, also in-repo at `slides/` (rebuild: `conda run -n torch colloquium build slides/recirculation.md`) — note: the live site root now serves the notebook, not the slides
- 📄 **Paper draft:** [`reports/paper_draft_recirculation.md`](reports/paper_draft_recirculation.md) · **Lead handover:** [`reports/research_handover_recirculation.md`](reports/research_handover_recirculation.md)

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

Two study lines (different protocols — do not mix numbers across them):

**A. Mozer replication (2026-09-23, current):** GSM8K-Platinum (n=1209),
Gemma-3-1B-IT, repo-parity protocol (multiturn `cot_llama` + chat template,
greedy, 256 tokens, FP16), path 25→20 α=0.04, batch 128, A100-40GB:

| condition | accuracy | Δ vs repo dense | McNemar p |
|---|---|---|---|
| Repo dense (taken, 540) | 0.4467 | — | — |
| Repo recirc (taken, 554) | 0.4582 | +0.0116 | 0.20 |
| **Ours mozer (531)** | **0.4392** | −0.0074 | 0.58 |
| **Ours cross-step (559)** | **0.4624** | +0.0157 | 0.25 |

Ours cross-step vs ours mozer (paired, same harness): +28 rows (+0.0232),
exact p=0.054 — borderline, not significant; cross-step is delayed
feedback, not the paper's recirculation, no novelty claimed.
Paper: [`reports/paper_gsm8k_mt_replication.md`](reports/paper_gsm8k_mt_replication.md).

**B. Phase-2 PT study (2026-09-22, superseded protocol):** full GSM8K test
(n=1319), greedy, two-stage Kojima prompting, pinned revisions:

| condition | accuracy | Δ vs base | McNemar p |
|---|---|---|---|
| 1B baseline | 0.0167 (22) | — | — |
| 1B + fixed recirc (11→4, α=0.15) | 0.0144 (19) | −0.0023 | 0.742 |
| 4B baseline | 0.2942 (388) | — | — |
| 4B + fixed recirc (18→9, α=0.15, β=1.0) | 0.2775 (366) | −0.0167 | 0.260 |

Plus: 4B diagnostic sweep (18 cells, n=100) with response surface and
heatmaps, a two-pass schedule panel, and the sweep-nominated cell
`a010_s18_d7` confirmed at full n=1319 — see below and
[`reports/phase2_analysis.ipynb`](reports/phase2_analysis.ipynb).
The replication study (repo-parity protocol, mozer vs cross-step,
Gemma quirks audit) lives in
[`reports/paper_gsm8k_mt_replication.md`](reports/paper_gsm8k_mt_replication.md)
with the lead handover in
[`reports/research_handover_recirculation.md`](reports/research_handover_recirculation.md).
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
`src/eval_harness/models/recirculation.py` (`cross_step`, `two_pass`, and
paper-exact `mozer` schedules behind one `schedule` flag) — evaluator,
tasks, prompts, parsers, scorers never change between conditions.
Multi-turn chat prompts flow as message lists through the same contract
(`build_messages`), rendered by each backend's chat template.

## Replication study (2026-09-23, Gemma-3-1B-IT / GSM8K-Platinum)

Repo-parity protocol: multiturn `cot_llama` 8-shot + chat template,
greedy, 256 tokens, FP16, repo stop strings, parser v1.1, path 25→20
α=0.04, batch 128 on A100-40GB (VRAM peak ≈20 GB). Dense baseline taken
from the reference artifacts (540/1209); no dense arm run here.

```bash
# Parity dry-run (no GPU): validates 17-message rendering, stops, parser
python scripts/evaluate.py --config configs/gsm8k_platinum_gemma3_1b_it_mozer_cotllamaMT_s25_d20_a004.yaml --dry-run

# Full arms on JarvisLabs A100 (auto-pause between runs)
bash scripts/jl_run.sh configs/gsm8k_platinum_gemma3_1b_it_mozer_cotllamaMT_s25_d20_a004.yaml --batch-size 128 --monitor 15 --pause
bash scripts/jl_run.sh configs/gsm8k_platinum_gemma3_1b_it_crossstep_cotllamaMT_s25_d20_a004.yaml --batch-size 128 --monitor 15 --pause
```

Parity audit (all fixed, all tested): multiturn-not-single-turn
prompting · model EOS set `[1, 106]` (not tokenizer id 1) · per-row
`position_ids` under left padding · repo-priority extraction (parser
v1.1) · Gemma sliding-window replay (`arm → crop(-1) → replay →
crop(0)`). Details: handover §7.

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
