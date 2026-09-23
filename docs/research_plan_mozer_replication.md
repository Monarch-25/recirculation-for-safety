# RESEARCH PLAN — Mozer (paper-exact) replication & fair schedule comparison on GSM8K-Platinum

**Created:** 2026-09-23 · **Status:** draft — start after code tasks land

> Overarching goal (project): inference-time deep-to-shallow Recirculation for **LLM safety** while preserving reasoning capability and benign utility. Capability reproduction is the **gate** before any safety claim (docs/README.md RQ1–RQ6).
>
> This plan's job: (1) implement the **paper-exact** `mozer` schedule, (2) replicate the ModelCloud repo's GSM8K-Platinum numbers on **`gemma-3-1b-it`**, (3) run a **prompting-matched fairness comparison** of our `cross_step` vs `two_pass` vs `mozer` vs `dense`. Safety probes come after the capability gate.

---

## 0. Context (one paragraph)

The paper method is **same-token, one-additional-iteration** feedback with **first-pass readout** (Eq. 1–2). The ModelCloud repo implements it and explicitly **withdrew** the "delayed cross-token intervention" (our `cross_step`) as recirculation evidence. Our `two_pass` is same-token but reads out the wrong (second) iteration. Therefore a paper-faithful `mozer` schedule is the prerequisite for any meaningful comparison. See `docs/findings_mozer_comparison.md` for full evidence.

---

## 1. Code milestones (do first; unit-tested, no GPUs needed)

| # | Task | Files | Done-when |
|---|---|---|---|
| C1 | Add `schedule: mozer` (validation + docs) | `src/eval_harness/core/config.py` | `from_dict` accepts it; `to_dict` round-trips |
| C2 | `mozer` generation: same-token replay, **first-pass readout**; convex-coupled `β_t = 1 − α_t` under ramp (paper default) | `src/eval_harness/models/recirculation.py` | reuse two-pass replay plumbing; readout = pass-1 logits; `recurrence_variant == "mozer_tokenwise_serial"` |
| C3 | 8-shot `gsm8k_cot_llama_v1` template (Evalution `cot_llama` plain format, exact join) | `src/eval_harness/prompting/chat.py` | byte-identical to `_build_plain_prompt` render |
| C4 | GSM8K-Platinum wiring (dataset-id override + distinct example-id prefix) | `src/eval_harness/tasks/gsm8k.py` | loads `madrylab/gsm8k-platinum main/test` (1209 rows) |
| C5 | Tests: unit (mixing, β-coupling, ramp, config) + integration (mozer α=0 ≡ baseline; determinism; metadata; differs from two_pass readout on real model) | `tests/unit/test_recirculation.py`, `tests/integration/test_recirculation.py` | `pytest -q` green (128+ suite) |

## 2. Replication run on JL — `gemma-3-1b-it`, GSM8K-Platinum

Match the repo protocol as closely as our harness allows (FP16→**bf16** is the one acceptable divergence; note it in every manifest).

**Config set** (add to `configs/`, plus `_jl.yaml` batch-64 variants):
1. `gemma3_1b_it_platinum_dense_cotllama` — intervention none
2. `gemma3_1b_it_platinum_mozer_cotllama_s25_d20_a004` — schedule `mozer`, source 25 → dest 20, α=0.04 (repo's paper-corpus-selected path)
3. `gemma3_1b_it_platinum_mozer_cotllama_s11_d4_a010` — **paper** 1B pair {11,4}, α=0.10 (pre-registered secondary)
4. `gemma3_1b_it_platinum_mozer_cotllama_s25_d20_a004_ramp10` — ramp_tokens=10 (paper's 1B ramp; repo default is off — decide this before running, keep both cells honest)

All: `prompt.template_name: gsm8k_cot_llama_v1`, `use_chat_template: true` (matches repo on IT), greedy, max_new_tokens 256, n=1209, `wandb_project: recirc-gsm8k`.

**Replication gate:** dense ≈ 540/1209 and mozer ≈ 554/1209 (i.e., repo's +14, +2.59%). Within ±0.5 pp of their published numbers = pass. If off, check: model revision (pin `google/gemma-3-1b-it` commit), tokenizer chat template, stop-token handling, max token budget, scorer.

## 3. Fair schedule comparison (same prompt, dataset, model)

Matched matrix on **Gemma 3 1B IT**, `gsm8k_cot_llama_v1` + chat template, n=1209:

| Condition | Schedule | Path / α | Question answered |
|---|---|---|---|
| dense | none | — | reference baseline (repo parity) |
| cross_step | cross_step | 25→20 @0.04 | our prior method under *their* prompting (fairness) |
| two_pass | two_pass | 25→20 @0.04 | second-pass-readout variant |
| **mozer** | mozer | 25→20 @0.04 | paper-exact (primary) |
| mozer (paper path) | mozer | 11→4 @0.10 | paper hyperparameter under our scorer |

Optionally repeat the 4-cell comparison on **Gemma 3 4B PT** using the paper 4B path **{18,9} β=1 nonconvex** and our prior cross-step best **{18,7}** to bridge to the already-completed n=1319 GSM8K work — but 4B is compute ×~6; schedule behind 1B.

**Pre-registered comparisons (paired, per-example, same example_id across conditions):**
- mozer vs dense (primary effect; expect near repo +14)
- cross_step vs mozer (does our withdrawn variant differ at same prompt?)
- cross_step vs two_pass (readout-iteration ablation)
- transition matrices: correct→wrong / wrong→correct / parse-failure counts per condition

## 4. Safety probes (only after gate passes) — Phase 3 extension

On **IT** models (where refusal behavior exists):
- TruthfulQA (truthfulness) + a refusal probe (e.g., AdvBench subset) dense vs mozer, same 8-shot-free / minimal prompting, n≈300.
- Benign over-refusal check (XSTest) to detect blanket-refusal artifacts.
- Report separately: harmful-refusal / benign-refusal / truthfulness / GSM8K capability / latency+memory — never a single "safety score".
- Then scale to the full Phase 3 battery (HarmBench, AdvBench, Sorry-Bench, WildJailbreak, XSTest, OR-Bench) per `docs/research_plan_phase3.md`.

## 5. Analysis & reporting

- Paired flips + transition tables per condition; attention-sink / source-norm diagnostics (extend `last_debug` capture to mozer).
- WandB: one project, condition tagged; run ids carry `_mozer_`, `_cross_`, `_twopass_`, `_dense_`, `_it_`, `_platinum_` suffixes.
- Results dirs per run; `scripts/compare_runs.py` for mozer-vs-dense and mozer-vs-cross.
- Commit code+configs; update `reports/`, deck, `site/` → gh-pages.

## 6. Compute / time budget (JL A100-40GB approximated)

- 1B serial prefill ~1.5–2× dense cost; mozer/two_pass ~2× dense; cross_step ~1–2×.
- Estimated per 1B full run (1209 rows, batch 64): **20–45 min**. Total Phase 2+3 matrix ≈ 8–10 runs ≈ **3–6 h** ≈ well inside the ₹25,029 JL grant (leave instance paused between runs; ~₹22/run at container rate).
- Dense parallelisable batching cuts the 1B dense arm down significantly.

## 7. Open questions to resolve before running

1. bf16 vs repo's FP16 (accept bf16; record in manifest) — acceptable?
2. `max_new_tokens` and stop-tokens: repo `until` list vs our fixed 256 — GSM8K answers fit; verify zero truncation on a smoke run.
3. `ramp_tokens`: repo default 0 but paper uses N=10 for 1B — run both cells pre-registered.
4. Scorer parity: adopt repo's `format_insensitive` priority (`final answer is < NUMBER >` before last-number)? Recommend **no** change now (last-number parses the contract correctly); revisit if replication misses.
5. Model revision pin for `google/gemma-3-1b-it` (resolve via HF API at config time like other models).

## 8. Acceptance for "Phase complete"

- [ ] `mozer` code + tests merged; suite green
- [ ] Gemma 3 1B IT dense ≈ 540/1209 and mozer ≈ 554/1209 on GSM8K-Platinum (≤0.5 pp drift)
- [ ] Full matched comparison matrix complete, all runs on WandB with artifacts
- [ ] Analysis report written (paired transitions, readout-iteration ablation)
- [ ] Phase 3 safety probe started (TruthfulQA + AdvBench subset + XSTest)