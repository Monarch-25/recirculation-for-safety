# Recirculation for LLM Reasoning: Phase-2 Handover Report

| | |
|---|---|
| **Status** | DRAFT v0.2 — for research-lead review (supersedes 2026-09-22 v0.1) |
| **Date** | 2026-09-23 |
| **Phase** | Phase 2 (capability replication on GSM8K) + Mozer replication |
| **Question for Phase 3** | Do recirculation's trajectory effects carry over to LLM safety? |
| **Quick links** | Replication paper: `reports/paper_gsm8k_mt_replication.md` · Draft: `reports/paper_draft_recirculation.md` · Tracking: W&B `recirc-gsm8k` |

## Contents

1. [Executive summary](#1-executive-summary)
2. [What changed since v0.1](#2-what-changed-since-v01)
3. [Background](#3-background)
4. [Methodology](#4-methodology)
5. [Research vitals](#5-research-vitals)
6. [Detailed findings](#6-detailed-findings)
7. [Gemma quirks log](#7-gemma-quirks-log)
8. [Discussion](#8-discussion)
9. [Recommendations (incl. Phase 3 + compute budget)](#9-recommendations)
10. [References](#10-references)
11. [Appendices](#appendices)

## 1. Executive summary

**Bottom line:** three evidence blocks, one honest picture. **(A)** Our
paper-exact Mozer replication on GSM8K-Platinum (Gemma 3 1B IT, reference
protocol, n=1209) scores **531 (43.92%)** — statistically indistinguishable
from the reference dense 540 (p=0.58) and recirc 554 (p=0.12):
**replication within noise.** **(B)** Our cross-step schedule — a
*different, withdrawn-class, non-novel* intervention — scores **559
(46.24%)** on the identical protocol: +28 paired rows over mozer at
**exact p=0.054 (borderline, not significant)**; and at its
sweep-nominated 4B cell (18→7, α=0.10) it scores **461/1319 (34.95%)**
against a 388 baseline (**+73, unpaired p≈0.002**). **(C)** The Bonferroni
caution from v0.1 graduated properly: the nominated cell was tested at
full scale and held up.

**Recommendation:** approve Phase 3 reframed around proven trajectory
effects, carrying **both** schedules (cross-step is now a legitimate
cost-effective control with full-scale evidence, measured at ~half the
recirc-arm compute). Approve a dev-split confirmation of the 4B cell
before any claim leans on it. Do not approve novelty language for
cross-step, and do not fund blind 12B scale-up. Budget ask: §9.3.

## 2. What changed since v0.1

1. **New headline numbers replace the old vitals.** The v0.1 table (1B
   22→19, p=0.74; 4B 388→366, p=0.26) used a non-parity harness and, for
   4B, a different cell (18→9, α=0.15). Current numbers are §5.
2. **Replication achieved.** Four parity gaps closed (multiturn prompting,
   model EOS set, per-row positions, extraction priority); see §7.
3. **Cross-step characterized honestly.** It differs from the paper's
   method (previous-token vs same-token source, mixed-run readout) and
   measures slightly above mozer without significance and without novelty.
4. **Sweep cell confirmed at scale.** `a010_s18_d7` (38/100, nominal
   p=0.012, Bonferroni-n.s.) → 461/1319 full-scale confirmation.
5. **Third schedule added.** `mozer` (paper-exact) joins `cross_step` and
   `two_pass` behind the same `schedule` flag; Gemma sliding-window
   replay handling fixed en route.

## 3. Background

Mozer et al. (arXiv:2608.17981): per-token first-iteration readout, then
same-token deep→shallow mixing with dest-L2 renormalization, upper-stack
replay replacing that token's upper KV, no second readout — recurrence in
depth and input step, distinct from looping. Reference headlines
(adaptive): −23% perplexity, +21% GSM8K. Reference fixed-GSM8K evidence on
GSM8K-Platinum/1B IT (`cot_llama` + chat template, 25→20, α=0.04): dense
540/1209, recirc 554/1209. The reference repo withdrew its early delayed
cross-token implementation as non-recirculation evidence — the class our
cross-step belongs to, labeled as such throughout.

![Paper: recirculation (top) vs looping (bottom) across scales](figures/paper_looping_vs_recirc.png)
*Paper Figure 8 (CC BY-NC-SA 4.0, Mozer et al.): the mechanism-class claim
we replicate (fixed-variant slice) and extend with a withdrawn-class
control.*

## 4. Methodology

### 4.1 Harness (unchanged architecture, extended parity)

`Task` owns semantics; `ModelAdapter` owns generation; `Evaluator`
orchestrates; baseline is first-class. New since v0.1: multi-turn message
prompts through the tokenizer chat template (`build_messages`,
17-message `cot_llama` layout); per-row `position_ids` under left
padding; model-EOS-set stopping (`[1, 106]` for Gemma 3 IT) plus repo
stop strings; parser v1.1 (repo extraction priority, Decimal
canonicalization); FP16 parity configs; batch-128 serial loop (VRAM peak
≈20 GB/40 GB). Acceptance gates still green (α=0 bitwise, batch
invariance, identical reruns; 124 unit tests + pre-existing env-gated
wandb skip).

### 4.2 Protocols (never mixed)

- **P-REP** (replication line): Platinum n=1209, multiturn `cot_llama` +
  chat template, greedy 256, FP16, parser v1.1, model rev `dcc83ea841ab`.
- **P2-PT** (prior line): `openai/gsm8k` n=1319, two-stage Kojima, PT
  checkpoints, v1.0 parser.

### 4.3 Schedules

- **Mozer** (paper-exact): same-token replay, first-pass readout (~2×
  forwards/token; measured 1296 s full run).
- **Cross-step** (withdrawn class): previous-token source, mixed-run
  readout, one forward/token (measured 669 s full run — ~half).
- **Two-pass** (v0.1 panel): same-token replay, rerun readout (n=100
  panel only).

### 4.4 Statistics

Exact two-sided McNemar (binomial) wherever paired predictions exist;
Wilson CIs; Bonferroni over sweep cells with nominal p-values; unpaired
two-proportion z-test only for the 4B full confirmation (archived
baseline lacks matching per-row verdicts — stated in every table).

## 5. Research vitals

| condition | acc | correct | 95% CI | Δ | p |
|---|---|---|---|---|---|
| Repo dense Platinum 1B IT (taken) | 0.4467 | 540/1209 | [41.9, 47.5] | — | — |
| Repo recirc Platinum 1B IT (taken) | 0.4582 | 554/1209 | [43.0, 48.6] | +0.0116 | 0.20 |
| **Ours mozer Platinum 1B IT** | **0.4392** | **531/1209** | [41.1, 46.7] | −0.0074 | 0.58 |
| **Ours cross-step Platinum 1B IT** | **0.4624** | **559/1209** | [43.4, 49.1] | +0.0157 | 0.25 |
| Ours cross vs ours mozer (paired) | — | +28 rows | — | +0.0232 | **0.054** |
| Ours cross-step 4B PT full (a010_s18_d7) | 0.3495 | 461/1319 | [32.4, 37.6] | +0.0553 vs 388 base | ≈0.002 (unpaired) |
| Prior: 1B PT (P2-PT) | 0.0144 | 19/1319 | [0.0092, 0.0224] | −0.0023 | 0.74 |
| Prior: 4B PT 18→9/α=0.15 (P2-PT) | 0.2775 | 366/1319 | [0.2540, 0.3023] | −0.0167 | 0.26 |

Costs (A100-40GB): P-REP pair ≈ 0.55 GPU-h (mozer 1296 s + cross 669 s);
4B full ≈ 0.3 GPU-h; probes/audit ≈ 1 GPU-h; session total ≈ 2 GPU-h.
Cumulative project ≈ 10–11 GPU-h.

## 6. Detailed findings

**Evidence map:**

| block | method | n | headline |
|---|---|---|---|
| A. Replication (§6.1) | mozer vs repo artifacts | 1209 | within noise (p=0.58/0.12), 82% row agreement |
| B. Cross-step IT (§6.2) | cross-step vs mozer, same harness | 1209 | +28 rows, p=0.054 — suggestive, n.s. |
| C. Sweep confirmation (§6.3) | a010_s18_d7 full | 1319 | +73 rows, unpaired p≈0.002 |
| D. Prior context (§6.4) | P2-PT pairs + sweep + two-pass panel | 1319/100 | nulls + surface + schedule-indifference |

### 6.1 Replication: implementation confirmed

Ours-mozer lands inside the reference band on all three paired tests
with 82.4% row agreement against repo dense. Residual gaps (−9/−23) fit
checkpoint drift (their revision unrecorded), backend numerics (FA2/eager
vs SDPA), and engine differences — no structural mismatch survived the
§7 audit.

### 6.2 Cross-step on the shared protocol: slightly above, not significant

559 vs 531 with 112/84 discordants (p=0.054). Report as
null-with-a-hint. The schedule is cheaper (669 s vs 1296 s) and now has
full-scale trajectory evidence, which promotes it from "ablation" to
"legitimate control" — still not to "method" or "novelty."

### 6.3 Sweep cell confirmed at scale

`a010_s18_d7`: 38/100 (+0.13, nominal p=0.012, Bonferroni-n.s.) →
461/1319 (+0.055, unpaired p≈0.002). The cautionary tale graduated by
testing. Destination-7 beats destination-9 at every source; s18 row
hottest — the surface story is unchanged, now with a full-scale anchor.

### 6.4 Prior context (v0.1, retained)

Single-point nulls (1B p=0.74; 4B 18→9 p=0.26), 69–90% churn with
symmetric flips, two-pass within ±0.07 of cross-step twins. Non-parity
harness; different 4B cell. Context, not verdict.

### 6.5 Runtime behavior

Serial prefill dominates recirc cost. Batch-128 ceiling validated by
32→64→128 probe (peaks ≈3.6→6.7→16.5 GB single-turn, ≈20 GB multiturn):
~2× headroom on 40 GB. vLLM baselines remain far cheaper per token —
engine-vs-eager, not method-vs-method.

## 7. Gemma quirks log (replication hazards, all fixed)

1. **Multiturn default.** `fewshot_as_multiturn` resolves True under chat
   templates → 17-message conversation. Single-message packing scores
   ~35% on the same model. Always log resolved message count.
2. **Model EOS ≠ tokenizer EOS.** Gemma 3 IT: model `[1, 106]`,
   tokenizer scalar `1`. Stopping on the scalar never fires (99% hit the
   cap); use the model set.
3. **Left-padding positions.** Shared `cache_position` misplaces RoPE for
   shorter rows; pass per-row `position_ids` from the cumulative mask.
4. **Sliding-window replay.** Arm recording → forward → `crop(-1)` →
   replay → `crop(0)`; skipping the closing restrict overflows the
   window (513-vs-512 SDPA).
5. **Stop strings.** Repo `cot_llama` stops rarely fire for Gemma but are
   required for parity claims; implement per-row truncation.
6. **Silent drift.** Unrecorded checkpoint revisions and BF16-vs-FP16
   greedy flips both move numbers; pin dtype + revision in manifests.

## 8. Discussion

The project moved from "null with churn" (v0.1) to "replication within
noise + a suggestive, non-significant control" (v0.2) — strictly ahead:
confounds named and fixed, p-values exact, one sweep cell confirmed at
scale. What would change the verdicts: a dev-split confirmation of the 4B
cell with paired predictions (cheap, commissioned next); an in-harness
dense arm (closes the last baseline gap); 12B/adaptive/pass@128 (only
after the first two). What would not: more n=100 cells, more rimming of
p=0.054, novelty language for cross-step.

## 9. Recommendations

1. **Approve Phase 3** reframed on proven trajectory effects, carrying
   both schedules; cross-step as cost-effective control (≈half compute).
2. **Commission first:** dev-split 4B-cell confirmation with paired
   artifacts + in-harness dense arm. Gate safety claims on these.
3. **Language guardrails:** "delayed-feedback control," never
   "recirculation," never "novel," for cross-step. Mozer arm may be
   called "paper-exact replication (within noise)."
4. **No blind 12B scale-up** until (2) resolves.

### 9.3 Compute budget

Spent: ≈10–11 GPU-h cumulative (Phase 2 ≈8–9 incl. Modal; this session
≈2 on JarvisLabs A100-40GB). Phase 3 projection unchanged in structure:
per safety benchmark of size N on 4B ≈ N×0.2 s (vLLM base) + N×2.5–5 s
(serial recirc, schedule-dependent) + 20%: N=1,000 → ≈1–1.5 GPU-h per
condition; 3-condition × 2-benchmark matrix ≈ 8–10 GPU-h. Request
**25 GPU-h** headroom incl. repeats + 12B pilot. Mac-side costs remain
zero (offline suite, dry-runs).

## 10. References

- Mozer et al. arXiv:2608.17981v1 (2026).
- ModelCloud/Recirculation + ModelCloud/Evalution.
- MadryLab GSM8K-Platinum; Cobbe et al. (2021); Kojima et al. (2022); Gemma 3 TR (2025).
- Hinton & McClelland (1987); Fan et al. (2020); vLLM RFC #53401; Kwon et al. (2023); McNemar (1947); Wilson (1927).

## Appendices

**A. Run inventory.** P-REP: mozer `results_jl/r_18a429d2` (531), cross
`results_jl/r_e19fafad` (559); 4B full `results_jl/r_694fe8aa` (461);
superseded single-turn `results_jl/r_9190756b` (414), `r_b40317ad`
(430); probes `r_2cc43e5e`, `r_8dc9ad72`, `r_6cd0059a`, `r_84e1e305`,
`r_d67dc29b`, `r_db196294`, `r_eb06a473` (metrics in W&B). Reference
artifacts: `gemma3_1b_gsm8k_dense_fp16.json`,
`gemma3_1b_gsm8k_recirc_best_fp16.json`. Prior-protocol bundle:
`reports/data/`, `reports/phase2_analysis.ipynb`. All mirrored to W&B
`recirc-gsm8k`. Comparison bundles in `comparisons/` (regenerable).

**B. Protocols.** P-REP values: Appendix of
`reports/paper_gsm8k_mt_replication.md` §6. P2-PT frozen values:
`docs/phase2_gsm8k_protocol.md` (do-not-modify).

**C. Statistics.** Exact McNemar via integer binomial arithmetic
(`math.comb`, no scipy); Wilson CIs; unpaired z only where flagged.

**D. File map.** `src/eval_harness/` (harness + 3 schedules),
`configs/*cotllamaMT*` (parity configs), `scripts/` (evaluate,
compare_runs, sweep, jl_run.sh, inspect_run), `analysis/` + `reports/`,
`site/` (notebook), `slides/` (deck source + build).

**E. Glossary (updated).** *Mozer schedule*: paper-exact same-token
replay + first-pass readout. *Cross-step*: previous-token deep→current
shallow, mixed-run readout — withdrawn class, non-novel control.
*Two-pass*: same-token replay, rerun readout (panel only). *Parity
protocol (P-REP)*: the reference-matching evaluation setup. *Churn*,
*rescued/regressed* as before.

**F. Cross-step walkthrough.** `appendix_crossstep_walkthrough.md`
(mechanics illustration; scores nothing).

**G. v0.1 review appendix.** Retained below from the 2026-09-22 review
(figures H-1…H-4, flip anatomy, open items) as the record of what v0.1
established; verdict language there is superseded by §1/§5 above where
they conflict.
