# Findings: Mozer et al. "Recirculation" vs our implementation (ModelCloud/Recirculation)

**Date:** 2026-09-23
**Sources:** Paper [arXiv:2608.17981v1](https://arxiv.org/html/2608.17981v1) (Mozer, Siddiqui, Sawyer, Sanyal, Liu — Google DeepMind); reference reproduction [github.com/ModelCloud/Recirculation](https://github.com/ModelCloud/Recirculation); their evaluation toolkit [github.com/ModelCloud/Evalution](https://github.com/ModelCloud/Evalution).

---

## 1. The paper's exact method — summary

The method is **same-token, one-additional-iteration** deep-to-shallow feedback with **first-pass readout**.

Notation (Eq. 1): `z_{i,j,l}` = residual-stream output after layer `l` at *unrolling step* `i` for *input step* `j`.

```
z_{t+1,t,d} = α · f(z_{t,t,s} | d, t) + β · z_{t,t,d}          (Eq. 1)
f(z | d, t)  = (‖z_{t,t,d}‖₂ / ‖z‖₂) · z                       (Eq. 2, dest-L2 renormalization)
```

- The **first iteration** of token `t` (unrolling step `t`) computes `z_{t,t,s}` (source layer) and `z_{t,t,d}` (destination layer) and produces the **readout**.
- The **additional iteration** of token `t` (unrolling step `t+1`) starts from the mixed boundary `z_{t+1,t,d}` and replays blocks `d+1..N`, **replacing token `t`'s upper-stack KV entries**.
- **Readout is taken only from the first iteration** — "the read out occurs following the first iteration of a stack" (Fig. 3 caption). The additional iteration "updates state but has no readout." This is the single most important structural fact.
- Because the replaced KV is attended to by tokens `t+1..`, influence propagates to subsequent tokens purely through attention state — a "recurrent state across token steps."
- Prefill is **serial** (cannot be parallelized).
- Default mixture is **convex, β ≡ 1−α**. Appendix B.3: **Gemma 3 4B and 12B critically require non-convex β = 1**.
- Ramping: `α_t = α · min(t/N, 1)` with N = 10 (zero-based). Introduced only for **1B** (early-token recirculation harms 1B; no harm in 4B/12B).
- Paper-selected layer pairs (Gemma 3 perplexity sweep): **1B {11,4}, 4B {18,9}, 12B {35,16}**.
- Recirculation ≠ looping: looped transformers recur in depth only; recirculation recurs in depth *and* input step.

## 2. ModelCloud/Recirculation reference repo

- Implements exactly the same-token, one-additional-iteration variant, with Torch as numerical oracle plus MLX/CUDA ports.
- **Explicit withdrawal that matters to us:** *"Results produced before the same-token replay correction measured a different **delayed cross-token intervention** and are withdrawn as recirculation evidence."* A cross-token scheme (previous token's deep state fed into the current token) is **not** the paper's method.
- Corrected "same-token replay": each token replays its own upper stack, replacing its upper-layer KV; with zero feedback the replay reproduces ordinary serial inference.
- Ramp default `ramp_tokens=0`; `ramp_tokens=10` added as the paper's zero-based ten-step ramp. Convex default coupling `β_t = 1 − α_t` when `β` is not explicitly overridden; an explicit non-default `β` stays fixed while `α` ramps. Zero-source-norm edge case normalized source defined as 0.
- **Their reported results (full splits, FP16, `cot_llama` variant, chat template ON 8-shot, greedy):**

| Model | Benchmark | Dense | Recirc | Delta | Rel |
|---|---|---|---|---|---|
| Gemma 3 1B IT, path `25→20`, α=0.04 | GSM8K-Platinum | 540/1209 (44.67%) | **554/1209** (45.82%) | +14 / +1.16 | +2.59% |
| Gemma 3 1B IT, `25→20`, α=0.04 | MMLU-STEM | 1084/3153 | 1083/3153 | −1 | −0.09% |
| Gemma 3 1B IT, `25→20`, α=0.04 | MMLU-Hum | 1736/4705 | 1737/4705 | +1 | +0.06% |
| Llama 3.2 1B IT, path `10→1`, α=0.04 | GSM8K-Platinum | 588/1209 (48.64%) | **597/1209** (49.38%) | +9 / +0.74 | +1.53% |

- GSM8K-Platinum `main/test` = **1,209 rows** (`madrylab/gsm8k-platinum`, sha `e7624924…`; columns `question`, `answer` (`#### N`), `cleaning_status`).
- Llama GSM8K delta flagged provisional (two unseeded dense runs differed on 5 row scores).

## 3. What we actually implemented (three schedules)

| Schedule | Mechanism | Readout | Paper parity |
|---|---|---|---|
| `cross_step` (ours, prior headline) | **root cross-token**: feeds stored deep@t−1 into shallow boundary at t; KV stores post-mix upper states | mixed run at t | ❌ This is the repo's **withdrawn "delayed cross-token intervention"** |
| `two_pass` (ours) | **same-token**: pass 1 records `z_{t,t,s}`/`z_{t,t,d}`, pass 2 mixes same-step source and reruns upper stack, overwriting KV | ❌ readout from **pass 2** (second iteration) | Close, but readout is wrong vs paper |
| `mozer` (to add) | **same-token replay + first-pass readout** = paper Eq. 1–2 exactly | ✅ first-pass (first iteration) logits; KV replaced by replay | ✅ |

- Mixing math is already paper-exact: dest-L2 renormalization `f`, `α·f + β·d`, 0-based block indexing `source > destination`, ramp factor `min(t/N,1)`, explicit non-convex β (paper-mandated β=1 for 4B).
- **Delta to implement `mozer`:** in the two-pass path, keep the *pass-1* logits for readout while still performing the mixed replay that overwrites KV. One code fork.
- Additionally: paper-default convex mixtures couple `β_t = 1 − α_t` when ramping (ours kept `β` fixed while ramping `α`). Implement the coupling for paper-exact `mozer`.

## 4. Prompting: ours vs repo (`cot_llama` in Evalution)

| Aspect | Ours (Phase-2 runs) | Repo (`cot_llama` via Evalution 0.0.12) |
|---|---|---|
| Model | `gemma-3-4b-pt` (pretrained) | `gemma-3-1b-it` / `llama-3.2-1b-instruct` (instruction-tuned) |
| Prompt | Zero-shot Kojima: `Q: {q} A: Let's think step by step.` (single-stage, `gsm8k_kojima_v1`) | 8-shot: `Given the following problem, reason and give a final answer to the problem.\nProblem: {q}\nYour response should end with "The final answer is [answer]"…` |
| Few-shot | 0 | 8 fixed `_LLAMA_FEWSHOTS` (targets end "The final answer is N") |
| Chat template | OFF (`use_chat_template: false`) | ON for their reported numbers (`apply_chat_template=true`) |
| Decode | greedy, max 256 / extract 32 | greedy, max_new_tokens 256 |
| Stop tokens | n/a (fixed length) | `<\|eot_id\|>`, `<\|start_header_id\|>user…`, `Q:`, `</s>`, `<\|im_end\|>`, `</assistant>` |
| Answer extraction | `####` → `\boxed{}` → last number | format-insensitive numeric: `#### N` → last "answer/final answer is" line → boxed → last line → last number |

**Verdict: not equivalent.** Differences: shot count, prompt contract (`final answer is`), chat-template usage, model tuning type, and answer-extraction priority.

**Changes required to match the repo protocol:**
1. New versioned template `gsm8k_cot_llama_v1` reproducing their `_build_plain_prompt` join (`prompt_builder + " " + target + "\n\n"` × 8, then the question prompt).
2. For exact replication on IT models: `use_chat_template: true` (chat template). For PT models the chat template is meaningless — keep OFF and note it.
3. Optionally adopt their numeric-extract priority (`final answer is` before last-number) as a new parser version — otherwise outputs like trailing "… 18" still parse via last-number (fine for their intended "The final answer is N" contract).

## 5. Implication for our existing results

- Our headline **cross-step n=1319 = 0.3495** (GSM8K, 4B PT) is a **different intervention** from the paper's recirculation. It cannot be presented as reproducing the paper.
- The ModelCloud repo explicitly **withdraws** such delayed cross-token results as recirculation evidence.
- `two_pass` is the closest of our current variants but reads out the wrong iteration.
- Therefore: implement `mozer` (paper-exact), replicate the repo's numbers on `gemma-3-1b-it` + GSM8K-Platinum, then do a **fair cross-step vs two-pass vs mozer comparison under identical prompting** (same template, dataset, model, decode).

## 6. Artifacts / references

- Paper: https://arxiv.org/html/2608.17981v1 (v1, 18 Aug 2026)
- Repo: https://github.com/ModelCloud/Recirculation (Apache-2.0; test `tests/test_recirculation.py` verifies norm-ratio mixture, prohibits cross-token injection)
- Evalution GSM8K suite: `evalution/benchmarks/gsm8k_common.py` (variants `base/cot/cot_zeroshot/cot_llama`; `_LLAMA_FEWSHOTS`; `_build_plain_prompt` join), `evalution/scorers/gsm8k.py` (`extract_format_insensitive_numeric_answer`)
- Dataset: `madrylab/gsm8k-platinum` config `main`, split `test`, n=1209, revision sha `e762492455a1cf7967de89f05b6bef72fc713b66`
- Repo eval config: `configs/gsm8k-platinum-cot-llama.yaml`
- Our code: `src/eval_harness/models/recirculation.py`, `src/eval_harness/core/config.py`, `src/eval_harness/prompting/chat.py`, `src/eval_harness/tasks/gsm8k.py`

---

## 7. Outcome (2026-09-23) — full paper: `reports/paper_gsm8k_mt_replication.md`

Parity-harness full runs (multiturn `cot_llama`, FP16, greedy, stops,
parser v1.1, s25→d20 α=0.04, batch 128, model rev `dcc83ea841ab`):

| Arm | Correct / 1209 | McNemar p vs repo dense (540) |
|---|---|---|
| Ours mozer (`results_jl/r_18a429d2`) | 531 (43.92%) | 0.58 — replicates within noise |
| Ours cross-step (`results_jl/r_e19fafad`) | 559 (46.24%) | 0.25 |
| Repo recirc (their artifact) | 554 (45.82%) | 0.20 (their own +14 n.s.) |

Ours-cross vs ours-mozer paired: +28 rows, exact p=0.054 (borderline,
not significant). Four parity gaps found and fixed en route: (1)
single-turn vs multiturn prompting (their `fewshot_as_multiturn`
defaults True → 17-message conversation; our single user message scored
~35%); (2) tokenizer EOS id 1 vs model EOS set [1, 106] (missed
`<end_of_turn>`, 99% hit the token cap); (3) scalar `cache_position`
misplacing RoPE under left padding (now per-row `position_ids`);
(4) last-number extraction vs their answer-line priority (parser v1.1,
~+8 rows). Gemma sliding-window replay additionally needs
arm → pass1 → `crop(-1)` → pass2 → `crop(0)` or SDPA overflows the
window. Verdict: mozer replicates; cross-step is delayed feedback, not
recirculation, with no significance and no novelty claim.