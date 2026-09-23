# Recirculation on GSM8K-Platinum: an independent replication (Gemma 3 1B IT) and a fair cross-step comparison

**Date:** 2026-09-23 · **Model:** `google/gemma-3-1b-it` rev `dcc83ea841ab`
**Benchmark:** GSM8K-Platinum (`madrylab/gsm8k-platinum`, `main/test`, n=1209, rev `e7624924`)
**Protocol:** Evalution `cot_llama` + chat template + multiturn 8-shot, greedy, 256 tokens, FP16
**Runs:** `results_jl/r_18a429d2` (mozer), `results_jl/r_e19fafad` (cross-step) · W&B `recirc-gsm8k`

## Abstract

We replicate the ModelCloud/Recirculation evaluation of Mozer et al.'s
training-free deep-to-shallow recirculation on GSM8K-Platinum with Gemma 3 1B
IT (path 25→20, α=0.04), and compare it — under an identical harness — to
our delayed cross-token ("cross-step") variant at the same path/alpha. Our
paper-exact mozer arm scores **531/1209 (43.92%)**, statistically
indistinguishable from the repo's dense baseline **540/1209 (44.67%,
McNemar exact p=0.58)** and its recirculation arm **554/1209 (45.82%,
p=0.12)**. Our cross-step arm scores **559/1209 (46.24%)**: +28 rows over
our mozer arm paired (**exact p=0.054**, borderline, not significant at
α=0.05). None of the GSM8K deltas in this study — ours or the repo's
(+14, p=0.20) — reach significance. Getting to parity required fixing four
implementation divergences (multiturn prompting, model EOS set, per-row
positions, extraction priority); the audit is documented so the next
replication does not repeat it. Cross-step is **not** the paper's method
(withdrawn intervention class) and is **not** claimed as novel.

## 1. What the paper's method is (one paragraph)

Per token `t`, Mozer et al. run the stack once (first iteration), read the
logits from that pass, then mix the same token's deep source residual into
its shallow destination boundary,
`z_{t+1,t,d} = α·f(z_{t,t,s}) + β·z_{t,t,d}` with dest-L2 renormalization
`f`, replay blocks `d+1..N`, and **replace token `t`'s upper-stack KV
entries without any second readout**. Influence reaches later tokens only
through attention over replaced KV — recurrence in depth *and* input step.
The reference reproduction (ModelCloud/Recirculation) explicitly withdrew
its earlier delayed cross-token results (*"a different delayed cross-token
intervention"*) as recirculation evidence after correcting to same-token
replay. Our `cross_step` schedule *is* that withdrawn class
(deep@t−1 → shallow@t, readout from the mixed run); our `mozer` schedule
is the paper-exact variant (same-token replay, first-pass readout).

## 2. Method

### 2.1 Parity protocol (ours vs repo)

| Axis | Repo (reported numbers) | Ours (this study) |
|---|---|---|
| Model | `gemma-3-1b-it` (local ckpt, rev unrecorded) | same id, Hub rev `dcc83ea841ab` |
| Prompt | `cot_llama` 8-shot, `apply_chat_template=true`, **multiturn** (17 msgs) | same (`gsm8k_cot_llama_multiturn_v1`), verified 17 msgs |
| Decode | greedy, temp 0, max 256, stop list, EOS stop | same; model EOS set `[1, 106]` |
| dtype | FP16 | FP16 (`float16`) |
| Attention | eager (dense) / FA2 paged (recirc) | default SDPA serial loop |
| Extraction | `####` → answer-line → boxed → last-line → last-number, Decimal compare | same priority (`gsm8k_parser v1.1`) |
| Batching | continuous, batch 8 / 16k-token budget | chunked serial, batch 128 |

### 2.2 Our mozer implementation

`src/eval_harness/models/recirculation.py`: hook-intercepted destination
boundary (output of block `d`), source capture (output of block `s`),
dest-L2 rescale, convex `β=1−α`, serial prefill + decode. Three fixes were
required for Gemma 3 correctness, all validated on-device:

1. **Sliding-window replay rollback.** Gemma 3's hybrid cache raises on
   `crop()` past the 512 window unless recording is armed. Per redo step:
   arm → pass 1 → `crop(-1)` (remove exactly the added token) → pass 2 →
   `crop(0)` (re-restrict to window). Without the closing restrict, the
   cache grows to window+1 and SDPA fails (513-vs-512).
2. **Per-row positions.** Batches are left-padded; the shared scalar
   `cache_position` misplaces RoPE for shorter rows. We pass explicit
   per-row `position_ids` from the cumulative mask (+1.3 pts on the
   single-turn full run: 414→430).
3. **Model EOS set.** The tokenizer scalar EOS is 1 (`</s>`); Gemma 3 IT
   ends turns with 106 (`<end_of_turn>`); model config lists `[1, 106]`.
   Stopping on id 1 alone left 99% of generations ramming the 256 cap.
   After the fix, mean output length is ~147 tokens, 12.5% hit the cap —
   matching the repo's EOS-terminated regime.

### 2.3 Cross-step arm (honest labeling)

Same harness, prompt, path/alpha, decode — only the schedule differs
(single forward per step, previous-token deep state mixed into the current
shallow boundary, readout from the mixed run). It is a delayed-feedback
baseline in the withdrawn class, **not** recirculation, and no novelty is
claimed (cf. feedback/recurrent-memory transformer literature; the
reference repo implemented and withdrew the same class).

### 2.4 Statistics

Primary: exact two-sided McNemar (binomial) on paired correctness;
supporting: Wilson 95% CIs. Dense baseline is the repo's 540/1209 (no
dense arm was run here by plan); index-aligned per-row artifacts
(`gemma3_1b_gsm8k_dense_fp16.json`, `..._recirc_best_fp16.json`) enable
exact paired tests against both repo arms. Order verified (target[0]=18,
n=1209 all arms).

## 3. Results

| Arm | Correct / 1209 | Accuracy | 95% Wilson CI | Δ vs repo dense | McNemar p vs repo dense |
|---|---|---|---|---|---|
| Repo dense (taken) | 540 | 44.67% | [41.9, 47.5] | — | — |
| Repo recirc (taken) | 554 | 45.82% | [43.0, 48.6] | +14 (+1.16) | 0.20 |
| **Ours mozer** | **531** | **43.92%** | [41.1, 46.7] | −9 (−0.74) | 0.58 |
| **Ours cross-step** | **559** | **46.24%** | [43.4, 49.1] | +19 (+1.57) | 0.25 |

Paired ours-cross vs ours-mozer (same harness, same prompt): 112
cross-only vs 84 mozer-only flips, net **+28 (+2.32 pts), exact p=0.054**.
Ours-mozer vs repo-recirc: −23, p=0.12. Per-row agreement ours-mozer /
repo-dense is 82.4%. All CIs overlap heavily.

Compute (A100-40GB, batch 128): mozer full 1296 s (21.6 min), cross-step
full 669 s (11.2 min, one forward per token vs two). VRAM peak
≈20.2 GB / 40 GB (51%), mean util ≈45%. Batch 128 was selected by a
32→64→128 scaling probe (peaks 3.6→6.7→16.5 GB single-turn; ≈20 GB
multiturn) as the safe ceiling with ~2× headroom. Parse rate 100% both
arms (0 failures).

## 4. Fair assessment

1. **Replication: success within noise.** Our mozer arm (531) lands inside
   the repo's dense/recirc band with p=0.58/0.12 and 82% row agreement,
   despite different revisions, attention backends, and engines. The
   remaining −9/−23 gaps are consistent with checkpoint drift (their
   revision is unrecorded) and backend numerics, not with a structural
   mismatch — the audit closed every structural gap we could find.
2. **Cross-step vs mozer: suggestive, not significant.** +28 paired flips
   at p=0.054 does not clear α=0.05 and was not corrected for multiple
   comparisons; it must be reported as a null-with-a-hint, not a win.
   It is also the wrong method to credit: cross-step is delayed
   cross-token feedback, explicitly not Mozer recirculation.
3. **No novelty claim.** Cross-step belongs to a known, previously
   implemented-and-withdrawn intervention class. Any future claim would
   require a literature review, a mechanism distinguishing it from prior
   feedback transformers, and a significant, replicated effect — none of
   which exist here.
4. **The repo's own +14 is not significant either (p=0.20).** The honest
   headline for GSM8K-Platinum at this operating point: neither
   intervention moves accuracy significantly; the study is powered
   (≈200–260 discordants) to detect roughly ±2.5 pts at 80% power.

## 5. Limitations

No in-harness dense arm (repo dense taken as baseline; residual harness
bias, if any, is shared by both our arms but not by the repo numbers).
Single model, single path/alpha (repo-selected 25→20, α=0.04), greedy
only. Stop-string edge semantics and skip-special-tokens decoding differ
cosmetically from the repo engine. Batch-128 serial loop, not continuous
batching (matched math, different performance envelope).

## 6. Reproducibility

Configs: `configs/gsm8k_platinum_gemma3_1b_it_{mozer,crossstep}_cotllamaMT_s25_d20_a004.yaml`
(code v0.4.0). Manifests pin model rev, FP16, template hash, parser v1.1,
6 stop strings, seed 42, transformers 5.17.0. Artifacts:
`results_jl/r_18a429d2` (mozer 531), `results_jl/r_e19fafad` (cross 559),
W&B project `recirc-gsm8k`. Superseded single-turn runs (≈34–36%) are
retained in `results_jl/` and labeled non-comparable (different prompt).

## References

- Mozer et al., *Recirculation*, arXiv:2608.17981v1.
- ModelCloud/Recirculation (reproduction + eval reports) and
  ModelCloud/Evalution (`gsm8k_common.py`, `scorers/gsm8k.py`).
- GSM8K-Platinum (`madrylab/gsm8k-platinum`).
