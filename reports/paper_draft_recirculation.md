# Inference-Time Deep-to-Shallow Feedback on GSM8K: Replicating Mozer Recirculation, Testing a Cross-Step Variant, and Confirming a Sweep Cell at Full Scale

> **Status:** draft v0.2 (2026-09-23) — for internal review. Supersedes draft
> v0.1 (2026-09-22). Author list TBD.
> **Method note:** every number below traces to a frozen artifact listed in
> Data Availability; nothing is reported from memory. Statistics are exact
> two-sided McNemar (binomial) on paired correctness plus Wilson 95% CIs.
> Companion documents: `reports/research_handover_recirculation.md` (lead
> handover), `reports/paper_gsm8k_mt_replication.md` (replication study
> paper), `reports/appendix_crossstep_walkthrough.md` (worked trace).

## Abstract

We study fixed, training-free deep-to-shallow feedback on frozen Gemma 3
checkpoints across three evidence blocks. **(A) Mozer replication:**
paper-exact recirculation (same-token replay, first-pass readout) on
GSM8K-Platinum with Gemma 3 1B IT under the reference protocol (multiturn
`cot_llama` + chat template, greedy, FP16, path 25→20, α=0.04) scores
**531/1209 (43.92%)**, statistically indistinguishable from the reference
dense baseline 540/1209 (44.67%, exact p=0.58) and the reference
recirculation arm 554/1209 (45.82%, p=0.12) — a replication within noise.
**(B) Cross-step variant:** our delayed cross-token schedule — a different,
withdrawn-class intervention, not the paper's method — scores **559/1209
(46.24%)** on the identical protocol: +28 paired rows over our mozer arm
(exact p=0.054, borderline, not significant at α=0.05), and the same
schedule at a sweep-selected cell (18→7, α=0.10, β=1.0) on Gemma 3 4B PT
scores **461/1319 (34.95%)** against a 388/1319 (29.42%) baseline: +73
rows (+5.53 pts, unpaired p≈0.002; paired test unavailable, see §4.2).
**(C) Sweep → full-scale confirmation:** the 18-cell n=100 diagnostic that
nominated cell `a010_s18_d7` (38/100, +0.13, nominal p=0.012,
Bonferroni-n.s. across 18 cells) held up when run at full n=1319 scale.
Getting to parity required fixing four implementation divergences
(multiturn prompting, model EOS set, per-row positions, extraction
priority) plus Gemma sliding-window replay handling — documented as a
replication audit (§3.5) so the next attempt does not repeat it. No
novelty is claimed for cross-step: it is previously
implemented-and-withdrawn delayed feedback, reported here as a measured
baseline with an honest p-value.

## 1. Introduction

**Research question.** Does deep-to-shallow representation recirculation —
an inference-time intervention that feeds a deep-layer activation back into
a shallower layer — change LLM reasoning behavior, and if so, do the
effects carry over to LLM safety behavior?

**What this paper is.** An independent replication and extension report on
the *fixed, training-free* variant only: no adaptive coefficients, no MLP
variant, no fine-tuning. Three evidence blocks: (A) a paper-exact Mozer
replication on GSM8K-Platinum/1B IT; (B) a same-harness comparison against
our delayed cross-token ("cross-step") schedule plus its full-scale 4B
confirmation at a sweep-selected cell; (C) the diagnostic sweep that
nominated that cell. Safety experiments remain out of scope; §6 states what
this work does and does not license for the safety phase.

**What changed since v0.1.** The v0.1 headline (1B: 22→19, p=0.74; 4B:
388→366, p=0.26, cross-step at the paper's single published point per
scale) stands as prior-protocol context but is superseded as the verdict:
the v0.1 runs used a non-parity harness (single-turn prompting, tokenizer
EOS, scalar positions) and, for 4B, a different cell (18→9, α=0.15) than
the sweep optimum. This draft reports the parity-harness replication, the
fair cross-step comparison, and the sweep cell's full-scale confirmation.

## 2. Background

Mozer et al. (arXiv:2608.17981) propose inference-time recurrence for frozen
transformers: per token, run the stack once, read out from that pass, mix
the same token's deep source into its shallow destination
(`z_{t+1,t,d} = α·f(z_{t,t,s}) + β·z_{t,t,d}`, dest-L2 renormalization `f`),
replay the upper stack, and replace the token's upper KV — with **no second
readout**. Influence propagates across positions purely through attention
state: recurrence in depth *and* input step, sharply distinguished from
looping (same-step depth recurrence). Reported headlines (adaptive,
Gemma 3): −23% perplexity, +21% GSM8K. For fixed recirculation on GSM8K:
greedy pass@1 gains on 4B PT at pairs {11→4} (1B) / {18→9} (4B), α=0.15,
convex β (1B) / β=1.0 (4B), 1B-only ramping. The reference reproduction
(ModelCloud/Recirculation) corrected an early delayed cross-token
implementation to same-token replay and explicitly withdrew the former
(*"a different delayed cross-token intervention"*) as recirculation
evidence; it reports, on GSM8K-Platinum/1B IT with 8-shot `cot_llama` +
chat template, dense 540/1209 and recirc 554/1209 at 25→20, α=0.04.

Our machinery: GSM8K (Cobbe et al., 2021) and GSM8K-Platinum (MadryLab) as
reasoning benchmarks, pinned open weights, paired binary statistics, and a
model-agnostic harness (`Task` owns semantics, `ModelAdapter` owns
generation) in which the baseline is a first-class condition.

### 2.1 Prior art and novelty boundaries

(1) The *term* "recirculation" is Hinton & McClelland (1987), a
weight-learning rule — unrelated mechanism, cited for the record.
(2) Inference-time deep→shallow feedback on frozen transformers is Mozer
et al. (2026), not re-claimed. (3) The closest *trained* family is the
Feedback Transformer (Fan et al., 2020) — same intuition, opposite regime.
(4) The community converged on the two-stack reading independently (vLLM
RFC #53401: normal pass + rerun-upper-stack, normal-pass readout).
**Cross-step as built here** — one forward per step, previous-token deep
state into current shallow boundary, readout from the mixed run — is the
withdrawn-class variant: the reference repo implemented and withdrew the
same class, and delayed-feedback architectures predate it. **No novelty is
claimed.** It is reported as a measured, honestly-labeled baseline.

## 3. Method

### 3.1 Harness and protocols

Two frozen protocols, never mixed: **P2-PT** (v0.1 line: `openai/gsm8k`
n=1319, two-stage Kojima, PT checkpoints, v1.0 parser) and **P-REP**
(this draft's replication line: `madrylab/gsm8k-platinum` n=1209,
multiturn `cot_llama` + chat template, greedy 256 tokens, FP16, parser
v1.1, 6 repo stop strings). Every run emits an immutable directory
(`manifest.json`, `config.yaml`, `metrics.json`, per-example
`predictions.jsonl`, `environment.json`, `logs.txt`) plus W&B mirroring
(`recirc-gsm8k`). Paired joins are exact on `example_id`.

### 3.2 The two schedules

- **Mozer** (paper-exact): pass 1 full stack, capture `h_s(t)`, `h_d(t)`,
  readout from pass 1; pass 2 mixes the same-step source, replays
  `d+1..N`, overwrites upper KV, no readout. ~2× forwards per token.
- **Cross-step** (withdrawn class): one forward per step; shallow boundary
  at `t` mixes stored deep source from `t−1`; KV stores post-mix states;
  readout from the mixed run. Position 0 warm-up.

Shared math: `α·f(s) + β·d`, dest-L2 rescale, convex/nonconvex β, 0-based
indexing with fail-fast bounds. **Acceptance gates:** α=0 ≡ baseline
bitwise; batch-size invariance; identical reruns; norm/ramp unit tests
(124 unit tests green; 1 pre-existing env-gated wandb failure).

### 3.3 Parity protocol (P-REP vs reference)

Model id, 8-shot pool, multiturn layout (17 messages), greedy/256/stops,
FP16, and extraction priority (`####` → answer-line → boxed → last-line →
last-number, Decimal compare) all mirror Evalution. Residual differences:
their checkpoint revision is unrecorded (ours `dcc83ea841ab`); attention
backend eager/FA2 vs our default SDPA serial loop; continuous batching vs
chunked serial batches (matched math, different performance envelope).

### 3.4 Statistics

Exact two-sided McNemar (binomial) on paired correctness wherever paired
predictions exist; Wilson 95% CIs; Bonferroni discipline over sweep cells
with nominal p-values reported. The 4B full confirmation (§4.2) reports an
unpaired two-proportion z-test because the archived baseline lacks
per-row verdicts in a matching schema — stated, not hidden.

### 3.5 Replication audit: four divergences plus Gemma quirks

1. **Single-turn vs multiturn.** Evalution's `fewshot_as_multiturn`
   defaults True under chat templates: 8 (user, assistant) turns + query.
   Our single user message scored ~35%; multiturn reaches the 44–46%
   regime. Template `gsm8k_cot_llama_multiturn_v1`.
2. **Tokenizer EOS vs model EOS.** Gemma 3 IT ends turns with
   `<end_of_turn>` (106); the tokenizer scalar EOS is 1. Model config
   lists `[1, 106]`. Stopping on id 1 alone left 99% of generations
   ramming the 256 cap; the model-EOS fix yields ~147-token means, 12.5%
   at cap — the reference's EOS-terminated regime.
3. **Scalar positions under left padding.** A shared `cache_position`
   misplaces RoPE for shorter rows; per-row `position_ids` from the
   cumulative mask (+1.3 pts on the single-turn full run).
4. **Extraction priority.** Last-number → repo priority (parser v1.1,
   ~+8 rows on our outputs).
5. **Gemma sliding-window replay.** The hybrid cache raises on `crop()`
   past window 512 unless recording is armed; the correct per-step
   protocol is arm → pass 1 → `crop(-1)` → pass 2 → `crop(0)`, where the
   closing restrict prevents window+1 SDPA overflow. Also recorded:
   FP16-vs-BF16 greedy flips exist (we match FP16); checkpoint revisions
   drift silently (pin everything).

## 4. Results

### 4.1 Replication block (P-REP, 1B IT, n=1209, 25→20 α=0.04)

| Arm | Correct | Accuracy | 95% CI | Δ vs repo dense | McNemar p |
|---|---|---|---|---|---|
| Repo dense (taken) | 540 | 44.67% | [41.9, 47.5] | — | — |
| Repo recirc (taken) | 554 | 45.82% | [43.0, 48.6] | +14 (+1.16) | 0.20 |
| Ours mozer | 531 | 43.92% | [41.1, 46.7] | −9 (−0.74) | 0.58 |
| Ours cross-step | 559 | 46.24% | [43.4, 49.1] | +19 (+1.57) | 0.25 |

Ours-cross vs ours-mozer paired (same harness, same prompt): 112
cross-only vs 84 mozer-only flips, net **+28 (+2.32 pts), exact p=0.054**
— borderline, not significant at α=0.05. Ours-mozer vs repo-recirc: −23,
p=0.12. Per-row agreement ours-mozer/repo-dense: 82.4%. Parse rate 100%
both arms. Compute (A100-40GB, batch 128, ~20 GB/40 GB peak): mozer full
1296 s, cross-step full 669 s (one vs two forwards per token).

### 4.2 Sweep-confirmation block (4B PT, kojima, 18→7 α=0.10 β=1.0)

The n=100 diagnostic nominated `a010_s18_d7` (38/100, +0.13 over its
25/100 baseline, nominal p=0.012, Bonferroni-n.s. across 18 cells — the
cautionary tale). Run at full n=1319: **461/1319 (34.95%)** vs baseline
388/1319 (29.42%): **+73 rows (+5.53 pts), unpaired p≈0.002**. The cell
held up and amplified. This is a *different cell* than the v0.1 headline
(18→9, α=0.15: 366, p=0.26 n.s.) — the response surface varies by cell,
exactly as the sweep mapped it (destination-7 beats destination-9 at
every source; s18 row hottest). No paired McNemar is reported here
(archived baseline lacks matching per-row verdicts); the unpaired test is
conservative. Batch 64, bf16, ≈980 s, VRAM peak ≈20.8 GB.

### 4.3 Prior-protocol context (P2-PT, v0.1, retained not repeated)

1B: 22→19 (p=0.74); 4B at 18→9/α=0.15: 388→366 (p=0.26); 69–90% output
churn with symmetric flips; two-pass panel within ±0.07 of cross-step
twins. These used the non-parity harness and a different 4B cell; they
stand as context, not as the verdict.

## 5. Discussion

**Interpretation.** (1) The Mozer replication lands inside the reference
band on every paired comparison — implementation confirmed, within noise.
(2) Cross-step measures slightly above mozer on the shared protocol (+28,
p=0.054) and confirms a sweep cell at 4B full scale (+73, unpaired
p≈0.002) — but the first is not significant and the second is unpaired,
so both stay *suggestive*, and cross-step remains the withdrawn class:
report it, do not crown it, do not call it recirculation. (3) Nobody's
GSM8K delta here clears significance — including the reference's own +14
(p=0.20). The Bonferroni caution from v0.1 now has its better estimate:
the nominated cell confirmed out-of-sample at full scale, which is how a
cautionary tale should graduate — by testing, not by reinterpreting.

**Alternative explanations, ranked:** checkpoint drift (their revision
unrecorded); attention-backend numerics (FA2/eager vs SDPA); engine
differences; statistical noise for small true effects. NOT supported:
wrong layers/alpha/beta/norm (verbatim), broken implementation
(invariants + parity audit), single-cell luck at 4B (sweep-nominated,
full-confirmed).

**Limitations.** No in-harness dense arm on P-REP (repo dense taken);
single path/alpha per study line; greedy only; unpaired 4B test;
stop-string/decode cosmetics differ; no 12B/adaptive/pass@128.

## 6. Implications for Phase 3

The safety question ("do trajectory effects transfer?") survives
unchanged and is now better instrumented: both schedules rewrite
trajectories at scale, cross-step costs ~half the recirc arm (one vs two
forwards per token — measured 669 s vs 1296 s full runs), and the parity
harness removes the confounds that clouded v0.1. Carry **both** schedules
forward (the cross-step arm is now a legitimate cost-effective control
with full-scale evidence, not merely an ablation), keep the utility
holdout (P-REP protocol verbatim), and do not fund blind 12B scale-up
until a dev-split confirmation of the 4B cell exists. Ask: **25 GPU-h**
headroom (unchanged basis, §9 of handover).

## 7. Conclusion

Paper-exact Mozer recirculation replicates within noise on GSM8K-Platinum;
our cross-step variant — a different, withdrawn-class, non-novel method —
measures slightly above it on the shared protocol without reaching
significance, and confirms its sweep-nominated 4B cell at full scale. The
project is ahead of v0.1: the confounds are named and fixed, the p-values
are exact, and the safety phase inherits machinery instead of mysteries.

## References

- Mozer, Siddiqui, Sawyer, Sanyal, Liu. *Recirculation.* arXiv:2608.17981v1 (2026).
- ModelCloud/Recirculation (reproduction + eval reports) and ModelCloud/Evalution (`gsm8k_common.py`, `scorers/gsm8k.py`).
- MadryLab GSM8K-Platinum; Cobbe et al. GSM8K (2021); Kojima et al. (2022); Gemma 3 Technical Report (2025).
- Hinton & McClelland (1987); Fan et al., Feedback Transformer (2020); vLLM RFC #53401; Kwon et al., PagedAttention (2023); McNemar (1947); Wilson (1927).

## Data Availability

P-REP runs: `results_jl/r_18a429d2` (mozer 531), `results_jl/r_e19fafad`
(cross 559); 4B full: `results_jl/r_694fe8aa` (461); reference artifacts
(`gemma3_1b_gsm8k_dense_fp16.json`, `..._recirc_best_fp16.json`); sweep
table `comparisons/sweep_4b_100/`; prior-protocol bundle
(`reports/data/`, `reports/phase2_analysis.ipynb`,
`analysis/gsm8k_phase2.py`); W&B `recirc-gsm8k`. Configs:
`configs/gsm8k_platinum_gemma3_1b_it_{mozer,crossstep}_cotllamaMT_s25_d20_a004.yaml`,
`configs/gsm8k_gemma3_4b_pt_recirc_a010_s18_d7.yaml`.
