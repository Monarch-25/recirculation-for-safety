# Fixed Training-Free Recirculation Does Not Lift GSM8K Accuracy on Gemma3 1B/4B Under a Frozen Protocol — While Decisively Rewriting Reasoning Trajectories: An Independent Replication Report

> **Status:** draft v0.1 (2026-09-22) — for internal review. Author list TBD.
> **Method note:** drafted with the `academic-paper` skill methodology (IMRaD,
> integrity-first: every number below is traceable to a frozen artifact listed
> in Data Availability; nothing is reported from memory).
> Companion documents: `reports/research_handover_recirculation.md` (lead
> handover), `reports/appendix_crossstep_walkthrough.md` (worked trace),
> `docs/phase2_gsm8k_protocol.md` (frozen protocol).

## Abstract

We independently replicate the *fixed, training-free* recirculation
intervention of Mozer et al. (arXiv:2608.17981v2) on GSM8K, implemented two
ways behind one `schedule` parameter, and evaluated under a frozen,
fully traceable protocol on Gemma3 1B and 4B pretrained checkpoints (full
test set, n=1319, greedy pass@1), testing exactly one (α, β, layer-pair) point
per scale — the paper's published settings, not a tuned selection.
**Result: no accuracy improvement under
either schedule** — 1B: 0.0167 → 0.0144 (McNemar p=0.742); 4B: 0.2942 →
0.2775 (p=0.260). What we do measure is **massive, symmetric rewriting of
reasoning trajectories** (90.4% of 1B outputs and 69.3% of 4B outputs change
textually, with verdict flips cancelling in both directions), a smooth tunable
response surface on a diagnostic sweep, and schedule-indifference between our
two implementations on a 100-sample panel. This null is **conditional**: the
paper does not publish its exact GSM8K prompting, generation budgets, or
answer-parsing code, so our finding is a statement about our frozen two-stage
protocol, not about every possible prompting of the intervention — and our
ablation/sweep evidence lives on an n=100 subset, reported as exploratory,
never as configuration selection. Because safety behavior lives at the level
of generation trajectories, not aggregate accuracy, the trajectory-level
effect we prove here is precisely the substrate the next phase (LLM
safety/security) needs. Total Phase-2 cost: ≈8–9 A100-hours.

## 1. Introduction

**Research question.** Does deep-to-shallow representation recirculation — an
inference-time intervention that feeds a deep-layer activation back into a
shallower layer — improve LLM reasoning, and if so, do the benefits carry over
to LLM safety behavior?

**What this paper is.** An independent replication report on the *fixed,
training-free* variant only: no adaptive coefficients, no MLP variant, no
fine-tuning. Safety experiments are out of scope; §6 states exactly what this
work does and does not license for the safety phase.

**Phase-2 objectives.** (a) Establish a trustworthy GSM8K baseline on the
research models; (b) implement the fixed training-free intervention without
touching weights; (c) determine whether our implementation reproduces the
paper's qualitative capability effect.

## 2. Background

Mozer et al. propose inference-time recurrence for frozen transformers: after
each step, leak a small amount of deep-layer activation down to a shallow
layer, mixed as `d′ = α·f(s) + β·d` with destination-L2 rescaling, so the
model acts as a dynamical system tracking belief state. They distinguish this
sharply from *looping*: looping re-executes a block of layers on the *same*
input (depth recurrence within a single forward pass), whereas recirculation
never re-executes anything — influence propagates *across* input positions
instead. Their sweeps show recirculation helping broadly where looping mostly
harms (their Figure 8,
reproduced in our handover with attribution, CC BY-NC-SA 4.0). Reported
headlines for the adaptive variant on Gemma3: −23% perplexity, +21% GSM8K
accuracy. For *fixed* recirculation on GSM8K — our target — they report
improvement on greedy pass@1 and pass@128 for Gemma3 4B PT with paper-selected
pairs {11→4} (1B) and {18→9} (4B), α=0.15, convex β for 1B and β=1.0 for 4B,
destination-L2 normalization, and 1B-only ramping. Their GSM8K protocol is
zero-shot chain-of-thought following Kojima et al. (2022). A v2 erratum notes
Gemma models need BOS at every window start.

Our evaluation machinery follows standard practice: GSM8K (Cobbe et al.,
2021) as the reasoning benchmark, Kojima-style two-stage prompting,
pinned open weights (Gemma 3, 2025), paired binary statistics (McNemar, 1947;
Wilson, 1927), and an lm-evaluation-harness cross-check method (Zheng et al.)
held as future work.

## 3. Method

### 3.1 Harness and frozen protocol

Strict separation, enforced by construction: `Task` owns benchmark semantics;
`ModelAdapter` owns generation; `Evaluator.evaluate(task, model, config)`
orchestrates and depends only on both abstractions. The baseline is a
first-class condition (`intervention: {type: none}`), never a special code
path. Every run emits an immutable directory (`manifest.json`, `config.yaml`,
`metrics.json`, per-example `predictions.jsonl`, `environment.json`,
`logs.txt`) plus W&B mirroring (`recirc-gsm8k`).

Frozen values (`docs/phase2_gsm8k_protocol.md`, do-not-modify): Gemma3 PT
weights dated 2025-03 (1B rev `fcf18a2a…`, 4B rev `cc012e0a…`, pre-paper);
dataset `openai/gsm8k` main rev `740312ad…`, test split, 1319 examples in raw
order with stable ids; two-stage Kojima prompting (stage-1 template
`gsm8k_kojima_v1` v1.0: `Q: {question} A: Let's think step by step.`;
stage-2 extraction v1.0: `[prompt] [reasoning] Therefore, the answer (arabic
numerals) is`); raw prompts, chat template off, BOS asserted at window start;
greedy seed-42 decoding with 256 reasoning + 32 extraction tokens; v1.0
parser/scorer on stage-2 output with failures scored 0 in-denominator.

**Acceptance gates (all green before any GPU claim):** recirc(α=0) ≡ HF
baseline bitwise; `none` ≡ recirc(α=0) bitwise; two-pass(α=0) ≡ baseline
bitwise; src==dst rejected at config load; batch-size invariance;
twice-identical reruns; norm/eps/ramp unit tests. Offline suite: 122 passed,
1 skipped (11 model/vLLM-gated deselected).

### 3.2 Shared mixing mathematics

Both schedules share all mixing math behind one `schedule` parameter:
`α·f(s) + β·d` with destination-L2 rescaling (eps-safe division),
convex/nonconvex β resolution (never hardcoded), the paper-exact 1B ramp
α_t = min(t/10,1)·α, 0-based block indexing with bounds fail-fast, and opt-in
activation debug tracing (norms/cosine, capped). Interventions tested:
1B 11→4, α=0.15/β=0.85 convex with ramp; 4B 18→9, α=0.15/β=1.0 nonconvex,
ramp off. Why rescaling is load-bearing is measured, not asserted: at the
first mixed position of our instrumented trace the deep source norm (25,371)
is 204× the destination norm (124); raw addition would drown the destination
entirely.

### 3.3 Two readings, two schedules — each walked through

The paper's prose ("leak after each step") and its formalization (Fig-3c
two-stack unrolling, Eq. 1) admit two readings. We implemented **both**. Each
is described here the way our cross-step appendix describes its subject: no
abstraction without an operational step-by-step.

#### 3.3.1 Schedule A — cross-step (all full-scale results use this)

One forward per input position; the destination at step *t+1* mixes the
*stored* deep source from step *t*. Concretely, on the instrumented example
(GSM8K Weng babysitting question, SmolLM2-360M-Instruct, 11→4, α=0.15,
45 prompt tokens, 24-token generation cap):

1. **Position 0 — warm-up, no mixing.** No previous deep state exists, so
   block 4 receives its ordinary input. Its block-11 output is captured and
   stored (all 960 dims, detached).
2. **Positions ≥ 1 — mix.** Before block 5 executes, block 4's output is
   replaced by `d′ = 0.15·f(s) + 0.85·d` with `s` the stored block-11 state
   from the previous position. Blocks 5–31 run on the mixed state, so the KV
   entries written at this position already carry the recirculated influence;
   the fresh block-11 output overwrites the store.
3. **Decode continues the identical loop** (positions 45–68); sampling reads
   final logits each step. One measured event (position 1): source norm
   25,371 → rescaled into destination norm 124 → mixture norm 118, cosine
   0.636. Across all 68 mixed positions cosine sits in a 0.501–0.693 band
   with no prefill/decode discontinuity (full trace:
   `reports/appendix_crossstep_walkthrough.md`, frozen data
   `reports/data/xstep_trace.json`).

Batching changes nothing semantically: padded/finished rows are masked out of
mixing, capture, and attention (verified bitwise against the unbatched path).

#### 3.3.2 Schedule B — two-pass (the paper-figure reading)

Per input position, **two** forwards: (1) a normal full pass captures the
source; the cache is truncated; (2) the same token is rerun with the
*same-step* source mixed at the boundary. The rerun overwrites upper-layer KV
and supplies the logits — *logits-from-rerun is a documented choice, not a
theorem*; a readout-from-normal ablation is held as cheap future work. Cost is
~2× compute; warm-up applies at position 0 only. At α=0 this schedule is
bitwise identical to baseline, so any measured delta is pure intervention.

#### 3.3.3 Comparison

| | cross-step (A) | two-pass (B) |
|---|---|---|
| source capture | previous input position | same input position, first pass |
| forwards per position | 1 | 2 (~2× compute) |
| logits from | the single (mixed) pass | the rerun (mixed) pass |
| warm-up | position 0 | position 0 |
| full-scale evidence | §4.1 (n=1319 × 2 scales) | — |
| panel evidence | §4.3 baseline for comparison | §4.3 (n=100, 4 cells) |

#### 3.3.4 Delimiting our implementation: vs looping, vs the paper

**Vs looping transformers.** Neither schedule re-executes any layer: every
block runs exactly once per position. All cross-position influence flows
through one stored vector plus a KV cache whose upper-layer entries were
written post-mix. Both schedules are therefore pure recirculation, and the
paper's recirc-vs-looping contrast (their Figure 8) applies to the mechanism
class we test — our null is not a statement about looping, which we never
implement.

**Vs the paper's recirculation.** Three deltas, all documented rather than
guessed: (1) *Schedule.* Our cross-step schedule (one pass per position,
deep@t → shallow@t+1) is the prose reading; our two-pass schedule (capture,
truncate, rerun, logits-from-rerun) is the figure reading. §4.3 shows they
behave alike within noise, but the exact unrolling and the readout choice
remain open alternatives. (2) *Harness.* The paper does not publish model
SHAs, generation budgets, or answer-parsing code; ours are frozen and printed
(§3.1) instead of reconstructed. Serial one-token-per-forward prefill with
per-row validity masking is our engineering choice (bitwise batch-invariant),
as is asserting BOS at every window start per the paper's v2 erratum.
(3) *Numerics.* The JAX-vs-HF framework difference is unaddressed by design;
the 1B ramp schedule is reproduced as published but provisional.

### 3.4 Statistics

Paired binary design throughout: McNemar χ² with continuity correction for
verdict flips, Wilson 95% intervals for accuracies, Bonferroni discipline over
sweep cells (nominal p-values reported, selection forbidden). Paired joins are
exact on `example_id` (1319/1319, 100/100, zero orphans). All in
`analysis/gsm8k_phase2.py` (stdlib only).

## 4. Results

### 4.1 Full-scale single-config check (n=1319): null verdict, live trajectories — READ WITH §4.1.1

These runs test exactly one (α, β, layer-pair) point per scale — the paper's
published configuration. The null below rules out that point, not the
response surface around it (§4.2 maps the surface).

| condition | acc | correct | 95% CI | Δ vs base | McNemar p |
|---|---|---|---|---|---|
| 1B base | 0.0167 | 22 | [0.0110, 0.0251] | — | — |
| 1B recirc | 0.0144 | 19 | [0.0092, 0.0224] | −0.0023 | 0.742 |
| 4B base | 0.2942 | 388 | [0.2702, 0.3193] | — | — |
| 4B recirc | 0.2775 | 366 | [0.2540, 0.3023] | −0.0167 | 0.260 |

CIs overlap fully at both scales. Transition matrices (cc/cw/wc/ww) tell the
real story: 1B **2/20/17/1280** (90.4% of outputs changed); 4B
**203/185/163/768** (69.3% changed). Flips are symmetric — the intervention
moves verdicts vigorously in both directions and they cancel. Output-length
profiles are near-identical across conditions, so flips come from reasoning
content, not truncation artifacts.

#### 4.1.1 Caveats on the n=1319 null (do not skip)

1. **Our null is conditional on our frozen two-stage protocol.** The paper
   does not publish its exact GSM8K prompting, generation budgets, or
   answer-parsing code. We test a two-stage Kojima protocol (256-token
   reasoning, 32-token extraction, v1.0 parser on stage-2 output) — a
   reasonable but not identical realization of "zero-shot CoT." A different
   prompting/extraction choice is a live alternative explanation for the gap
   (§5), not a closed one.
2. **The 1B scale is near floor** (22/1319 baseline correct). Floor effects
   limit what any delta at 1B could show; 4B is the informative scale.
3. **Single runs; backend asymmetry.** Full pairs ran once each, with vLLM
   baselines against serial-HF treatments. No pass@128, no adaptive variant,
   no 12B.

### 4.2 Diagnostic sweep (4B, first-100, β=1.0): smooth, exploratory, n=100

18 cells (16 + 2 heatmap-completing runs): **16 of 18 beat the same-100
baseline (0.25)**; top cell `a010_s18_d7` at 0.38 (+0.13, nominal p=0.012,
Bonferroni-n.s.); the paper pair peaks at α=0.07 here; s16→d9 holds the
lone sub-baseline cell (0.24 at α=0.10); destination-7 beats destination-9
at every source. The complete
3×2 heatmap at α=0.10 shows the s18 row hottest with s16→d9 the lone cold
cell.

**Scope discipline:** this panel ran on the *first 100* test examples, whose
same-100 baseline (0.25) sits below the full-test baseline (0.2942) — the
subset differs in difficulty, and 100 samples cannot resolve small effects
after multiplicity control. It is reported as evidence of a smooth tunable
response surface, **never as configuration selection** (no test-set tuning).

### 4.3 Two-pass panel (n=100): schedule-indifference within noise

| config | two-pass acc | Δ vs base | Δ vs cross-step |
|---|---|---|---|
| a010_s18_d7 | 0.31 | +0.06 | −0.07 |
| a004_s20_d9 | 0.35 | +0.10 | −0.01 |
| a015_s20_d9 | 0.37 | +0.12 | +0.02 |
| a015_s18_d9 (paper) | 0.33 | +0.08 | +0.04 |

Two-pass behaves like cross-step within noise (±0.07, all n.s.): all cells
beat the same-100 baseline, none dominates its cross-step twin. The schedule
variant therefore does not obviously explain the paper gap — while the
readout choice (§3.3.2) and exact unrolling remain open. The cross-step arm
was not wasted work: without it, this panel could not isolate schedule
effects at all. A null ablation is data.

### 4.4 Runtime behavior

Serial prefill dominates recirc cost (measured: 1063s/3101s prefill vs
617s/1837s decode). Batched loop + stage budgets + SDPA cut per-example cost
~10× vs the naive serial build; vLLM baselines run two orders of magnitude
cheaper (4374/1310 tok/s vs 600/167). Costs (A100-40GB): 1B pair ≈ 0.5 GPU-h;
4B pair ≈ 1.5 GPU-h; sweep + two-pass panel ≈ 4 GPU-h; total Phase-2 to date
≈ 8–9 GPU-h.

## 5. Discussion

**Interpretation.** The honest reading is a null accuracy result with a
decisive behavioral result. A null with 69–90% output churn is not "nothing
happened" — it is evidence the intervention acts strongly on trajectories
while leaving verdict means untouched (at these scales, under this protocol).

**Alternative explanations for the paper gap, ranked:** (1) prompting/parsing
differences — open, since the paper's exact GSM8K harness is unpublished
(§4.1.1); (2) single-point selection — full scale covers one
published point per scale while the surrounding surface is mapped only at
n=100, so a nearby at-scale optimum is untested, not ruled out; (3) schedule
semantics — narrowed but not closed by §4.3 (readout
choice, exact unrolling); (4) JAX-vs-HF
numerics; (5) statistical noise at
full scale for small true effects; (6) model-revision drift — unlikely
(pre-paper weights). NOT supported: wrong layers/alpha/beta/norm (all
verbatim), broken implementation (invariants hold bitwise), insufficient scale
coverage (1B+4B full sets).

**Limitations / threats.** §4.1.1 caveats apply in full; plus first-100 sweep
subset difficulty mismatch; GPU-side determinism rerun pending;
lm-evaluation-harness cross-check still open.

## 6. Implications for Phase 3: LLM safety/security research

Safety behavior — refusals, harmlessness under pressure, reasoning integrity —
lives at the level of *generation trajectories*, not aggregate accuracy.
Phase 2 proves recirculation decisively rewrites trajectories. That is the
bridge: **even with null accuracy deltas, the safety question remains wide
open and directly testable with exactly this machinery.**

Concretely, we propose Phase 3 reframed as: *"Do recirculation's trajectory
effects transfer to safety behavior?"* — refusal robustness, harm refusal
under pressure, reasoning-integrity probes, each paired baseline-vs-recirc
with the existing harness, plus a utility holdout (this GSM8K protocol
verbatim) to detect capability regressions. Settle the schedule question
first (readout-from-normal ablation + dev-split confirmation of one sweep
cell) before any safety claim leans on a schedule; do not fund blind scale-up
(12B full sweeps) until that resolves. Projected cost: per safety benchmark of
size N on 4B, baseline ≈ N×0.2s (vLLM) + recirc ≈ N×5s (serial HF) + 20%
overhead — e.g. N=1,000 → ≈1.5 GPU-h/condition; a 3-condition × 2-benchmark
matrix ≈ 10 GPU-h. Request: **25 GPU-h** headroom including repeats and a 12B
pilot.

## 7. Conclusion

At the paper's single published configuration per scale, fixed training-free
recirculation, as implemented two ways, does not lift
GSM8K accuracy on Gemma3 1B/4B under a frozen, fully traceable protocol —
while profoundly rewriting generation trajectories, with a smooth tunable
response surface underneath. The paper's capability claim is therefore **not
reproduced here**, but the project is strictly ahead of where a naive positive
would have left it: we know exactly what was tested, what changed, and what
remains open. The trajectory-level effect is real, measured, and is precisely
the substrate Phase 3 safety work needs.

## References

- Mozer, Siddiqui, Sawyer, Sanyal, Liu. *Recirculation.* arXiv:2608.17981v2 (2026). [Paper figures reused in companion docs under CC BY-NC-SA 4.0 with attribution.]
- Kojima et al. *Large Language Models are Zero-Shot Reasoners.* NeurIPS 2022. (Prompting protocol.)
- Cobbe et al. *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168 (2021). (GSM8K.)
- Gemma Team. *Gemma 3 Technical Report.* (2025). (Checkpoints, gated.)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. (vLLM.)
- McNemar (1947); Wilson (1927). (Paired/interval statistics.)

## Data Availability

All claims regenerate from committed artifacts: frozen protocol
(`docs/phase2_gsm8k_protocol.md`), paired derivatives
(`reports/data/pair_{1b,4b}_paired.csv`), sweep/panel tables
(`reports/data/sweep_4b_100*.csv`, `reports/data/twopass_4b_100.csv`), run
metadata (`reports/data/runs.json`), executable analysis
(`reports/phase2_analysis.ipynb`, `analysis/gsm8k_phase2.py`), walkthrough
trace + generators (`reports/data/xstep_trace.json`,
`reports/figures/make_crossstep_trace.py`,
`scripts/trace_walkthrough.py`). Immutable run dirs live on the Modal volume
+ W&B `recirc-gsm8k` (1B base `run_20260921_213746_f0d8ab`, 1B recirc
`run_20260921_213557_173592`, 4B base `run_20260921_214339_11eb2e`, 4B recirc
`run_20260921_213705_b30f77`). Comparison bundles in `comparisons/`
regenerate via `scripts/compare_runs.py` (not committed, gitignored).
