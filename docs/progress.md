# Progress vs `research_plan.md` — agent guide

> Purpose: single source of truth for "where are we, what's proven, what's next".
> Plan source: `docs/research_plan.md` (§1–§53). Last checked: 2026-09-21 (steps 1–3 re-verified this date).
> Working dir: `/Users/mozart/Documents/ml/research/recirculation`.

## 1. TL;DR

**The Phase-1 harness is implemented and has produced real 5-example GSM8K runs on the M2 Mac.**
All core deliverables (§52, items 1–18) exist. What remains is **verification + small metadata fixes**,
not new architecture. **Do NOT implement Recirculation, safety benchmarks, Modal, or distributed code.**

| Area | Status |
|---|---|
| Modular harness (evaluator / task / adapter / parser / scorer / artifacts) | ✅ Done, matches §4 layout |
| GSM8K + SmolLM2-360M-Instruct (<500M) on MPS | ✅ Done, 2 real runs, acc 0.2 @ n=5 |
| Determinism, batching, ordering, per-example JSONL, manifests | ✅ Done in code + mock tests |
| Docs (`README.md`, `docs/evaluation_protocol.md`) | ✅ Done |
| Acceptance evidence (§47 steps 1–6) | ✅ Done 2026-09-21 — pytest 56 passed/1 skipped, dry-run ok, 2 fresh n=5 runs, paired table clean |
| Git traceability | ✅ Fixed 2026-09-21 — repo init'd (`4eaffe0`), manifests carry real commit/branch/dirty |
| Minor metadata gaps | ✅ Fixed 2026-09-21 — `tokenizer_revision` resolves to model sha; `dirty=false` on clean trees (was `None`) |
| Future work (full 1319, lm-harness cross-check, Recirculation, Modal) | ⏳ Intentionally not started |

Existing runs (do not delete):
- `results/gsm8k/smollm2-360m-instruct/run_20260905_063204_7ae5c7/` (`--limit 5`, pre-git, null git fields)
- `results/gsm8k/smollm2-360m-instruct/run_20260905_064219_0f658c/` (`--limit 5 --batch-size 2`, pre-git)
- `results/gsm8k/smollm2-360m-instruct/run_20260921_171914_3e3edf/` (`--limit 5`, batch 1 — acceptance re-run)
- `results/gsm8k/smollm2-360m-instruct/run_20260921_172120_1cbda9/` (`--limit 5 --batch-size 2` — acceptance re-run)
- All n=5 runs score `acc=0.2, parse_rate=1.0`. The 2026-09-21 pair has real git fields,
  `tokenizer_revision == model revision == a10cc15…`, and byte-identical `raw_output` per `example_id`.

## 2. Section-by-section status

### Architecture & separation (§1–§5, §41–§44, §49–§51)
| Plan | Status | Evidence |
|---|---|---|
| §1 pipeline `Dataset→Task→Prompt→Adapter→Parse→Score→Artifacts` | ✅ | `src/eval_harness/core/evaluator.py:_run` orchestrates, no model/task conditionals |
| §3 `evaluator.evaluate(task, model, config)` | ✅ | `src/eval_harness/core/evaluator.py:154` |
| §4 repo structure | ✅ (+extras) | Matches plan; extras: `models/mock.py`, `models/vllm_adapter.py`, `utils/hub.py` — acceptable, keep |
| §5 `ModelAdapter.generate(prompts, config)` + `HFCausalLMAdapter` | ✅ | `src/eval_harness/core/interfaces.py:63`, `src/eval_harness/models/hf_causal_lm.py:63` |
| §41 no global state | ✅ | DI everywhere; `scripts/evaluate.py:build_task/build_model` |
| §42 no premature abstractions | ⚠️ deviation | Plan says NOT to build vLLM yet; `vllm_adapter.py` + `*_vllm.yaml` already exist. Keep (proves swappability), do NOT extend further (no Modal/Ray/Recirculation) |
| §43–44 recirculation compat (`intervention: {type: none}`) | ✅ | `src/eval_harness/core/config.py:36`, passthrough in both adapters |
| §49–51 code quality, `generate()`-not-`pipeline()`, correctness>speed | ✅ | Typed dataclasses, `eager` attention documented in `docs/evaluation_protocol.md` |

### Model / task / prompting / generation (§6–§12)
| Plan | Status | Evidence |
|---|---|---|
| §6 HF adapter: CPU/MPS/CUDA, dtype, batched, deterministic, revisions | ✅ | `src/eval_harness/models/hf_causal_lm.py:resolve_device`, left-padding, per-call seeding |
| §7 model+tokenizer revision pinning | ✅ code / ⚠️ data | Resolves Hub sha (`a10cc15…` in manifests); user pin respected. Gap: `tokenizer_revision` records `null` when unpinned — should default to model sha |
| §8–9 GSM8K task, `openai/gsm8k/main/test`, stable IDs, `limit` | ✅ | `src/eval_harness/tasks/gsm8k.py`; IDs `gsm8k-test-{idx:05d}-{sha8}` |
| §10 frozen prompt `gsm8k_cot_v1` + chat template + per-example save | ✅ | `src/eval_harness/prompting/chat.py:20`, `format_for_chat` shared by HF+vLLM |
| §11 `GenerationConfig`, greedy `t=0/do_sample=false/top_p=1`, `max_new_tokens` 512 (256 debug) | ✅ | `src/eval_harness/core/interfaces.py:27`, configs |
| §12 evaluator-owned batching, order-preserving | ✅ | `src/eval_harness/core/evaluator.py:222`, `_chunked_with_start` |

### Parsing / scoring / results (§13–§17, §23–§25, §40)
| Plan | Status | Evidence |
|---|---|---|
| §13 `GSM8KAnswerParser` versioned/deterministic | ✅ | `src/eval_harness/parsing/gsm8k.py` (#### → boxed → last-number) |
| §14 scorer exact-match + `accuracy/parse_rate/num_*`, failures in denominator | ✅ | `src/eval_harness/scoring/gsm8k.py`, `src/eval_harness/core/results.py:42` |
| §15 per-example JSONL with all 7 required keys | ✅ (superset) | `predictions.jsonl` has `example_id/prompt/raw_output/parsed_answer/reference_answer/parse_success/correct` + `index/parsed_reference/score` |
| §16 paired comparison via stable IDs | ✅ | `scripts/inspect_run.py:32` (cc/cw/wc/ww table) |
| §17 unique run dirs `results/<task>/<slug>/run_<ts>_<id>/` | ✅ | Both runs present; `Evaluator.make_run_dir` |
| §23–25 prompt hash, `task_version 1.0`, `evaluation_code_version 0.1.0`, `result_schema_version 1.0` | ✅ | Manifests confirm |
| §40 manifest schema | ✅ | `manifest.json` has all required blocks |

### Reproducibility (§18–§22)
| Plan | Status | Evidence |
|---|---|---|
| §18 mandatory list (git, python, torch, transformers, cuda, model/dataset/tokenizer rev, dtype, seed, gen, batch, prompt, task+code version) | ✅ (2026-09-21) | All present in fresh manifests; git real, tokenizer resolved. Code: `core/evaluator.py:build_manifest`, `runtime/environment.py`, `runtime/hardware.py`, `runtime/reproducibility.py` |
| §19 git commit/branch/dirty, never fail, warn on dirty | ✅ code + env (2026-09-21) | Commit `4eaffe0`, dirty correctly `true` with uncommitted changes. Also fixed a bug: clean trees reported `dirty=None` (empty porcelain mapped to None); now `false`. See `runtime/reproducibility.py:get_git_metadata` |
| §20–21 package + CUDA/device metadata, `null` when unavailable, no `nvidia-smi` assumption | ✅ | `environment.json` shows `mps/device`, `cuda_available:false`, `cuda_version:null`, `gpu_model:"Apple Apple M2"` |
| §22 single `collect_environment_metadata()` | ✅ | `src/eval_harness/runtime/environment.py:23` |

### Config / CLI / modes (§26–§29, §35–§36)
| Plan | Status | Evidence |
|---|---|---|
| §26 YAML → typed objects, adapter never sees YAML | ✅ | `src/eval_harness/core/config.py:load_config_from_dict`, 3 configs in `configs/` |
| §27 CLI `--config` + `--limit/--batch-size` overrides + header | ✅ | `scripts/evaluate.py:131`; extra `--device` is fine |
| §28 `--dry-run` (no generation) | ✅ code / ⏳ re-verify | `scripts/evaluate.py:100` |
| §29 `--limit 3` full-pipeline validation | ✅ code / ✅ ran at n=5 | Both result dirs prove it |
| §35 structured logging, no response dumps | ✅ | `logs.txt` per run + `utils/logging.py` |
| §36 loud config errors, per-example failures → `parse_success=false, correct=false` | ✅ | `evaluator.py:243` try/except + `test_parse_failure_does_not_crash` |

### Tests (§30–§33, §37–§39)
| Plan | Status | Evidence |
|---|---|---|
| §30 unit (config/prompt/hash/parse/score/env/manifest/serial/batching/IDs) | ✅ | `tests/unit/test_{config,parser,prompt,results_meta,scoring,vllm_adapter}.py` |
| §30–31 integration via `MockModelAdapter` + tiny fixture, no internet | ✅ | `tests/integration/test_evaluator_mock.py`, `models/mock.py`, `tests/fixtures/gsm8k_tiny.json` |
| §33 `pytest` offline fast; `-m model` / `-m vllm` separated | ✅ | `pyproject.toml:markers`; `test_model_smoke.py`, `test_vllm_smoke.py` |
| §37–39 ordering, determinism ×2, batch invariance 1/2/4 | ✅ (mock) | `test_ordering_preserved`, `test_determinism_twice_identical`, `test_batch_size_invariance` |

### Docs & acceptance (§45–§48, §52–§53)
| Plan | Status | Evidence |
|---|---|---|
| §45 lm-eval-harness cross-check | ⏳ documented, not run (correct) | `docs/evaluation_protocol.md:102` |
| §46 `README.md` + `docs/evaluation_protocol.md` contract | ✅ | Both exist; README covers install/test/dry-run/smoke/full/results |
| §47 steps 1–6 | ✅ (2026-09-21, see §3 below) | `pytest` 56 passed/1 skipped; dry-run ok; 2 fresh n=5 runs with all 6 files and 7-key rows; batch 1 vs 2 byte-identical |
| §48 CLI output | ✅ ~minor | Matches spec except header lacks pre-execution `Run ID` (ID is created inside `evaluate()` — acceptable, don't fake it) |
| §52 deliverables 1–18 | ✅ | All files present; #18 M2 commands in README |
| §53 swappability (HF→Recirculation, same task/prompts/scoring/schema) | ✅ by construction | HF + vLLM already swap with zero evaluator/task change |

## 3. What to do next (priority order — agent checklist)

**Do these in order. Stop after item 4 unless the user explicitly asks for scale-up.**

- [x] **1. Make the directory a git repo (fixes §19 traceability).** — Done by user 2026-09-21 (`4eaffe0 Phase-1 harness baseline`).
  ```bash
  git init && git add -A && git commit -m "Phase-1 harness baseline"
  python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --dry-run
  ```
  ~~Expect~~ New manifests carry real `git.commit/branch/dirty` instead of nulls. ✅ Confirmed.

- [x] **2. Fix `tokenizer_revision: null` (small, §7/§18).** — No edit needed: the fallback
  (`hf_causal_lm.py:116-119`, tokenizer defaults to resolved model sha) was already in the
  committed code; the Sept-5 nulls came from pre-commit code. Fresh manifests confirm
  `tokenizer_revision == revision == a10cc15…`. ✅ Verified, not changed.
  Bonus fix in the same area: `get_git_metadata()` reported `dirty=None` on clean trees
  (empty porcelain output was mapped to None, leaving a dead `status == ""` branch).
  Fixed in `src/eval_harness/runtime/reproducibility.py` — now `false` when clean,
  `true` when dirty, `None` only when git fails. **Uncommitted — needs a commit.**
  Offline tests re-run green after the edit (55 passed).

- [x] **3. Re-run the §47 acceptance chain cleanly (conda env `torch`).** — Done 2026-09-21:
  ```bash
  conda activate torch
  pytest -q                                  # ✅ 56 passed, 1 skipped (vLLM/Linux-only)
  python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --dry-run   # ✅ passed
  python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --limit 5    # ✅ run_20260921_171914_3e3edf, acc=0.2
  python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --limit 5 --batch-size 2  # ✅ run_20260921_172120_1cbda9, acc=0.2
  python scripts/inspect_run.py results/gsm8k/smollm2-360m-instruct/run_<A> results/gsm8k/smollm2-360m-instruct/run_<B>
  # ✅ paired 1 cc + 4 ww, zero off-diagonal; raw_output byte-identical per example_id
  ```
  Pass criteria met: both runs have all 6 files; every `predictions.jsonl` row has the 7 keys;
  `example_id→raw_output` mapping identical across batch sizes. (Accuracy 0.2 is a model
  result, not a harness failure.)

- [ ] **4. Optional tidy (only if trivial): record `ram_gb` on macOS.**
  `runtime/environment.py:_ram_gb` returns `None` on Darwin despite the `sysctl` fallback — check why and fix or document as known gap. Do not block on this.

- [ ] **5. Do NOT do yet (explicit plan constraints):** full 1319-example run, `lm-evaluation-harness` cross-check (§45), `RecirculationModelAdapter`, safety tasks, Modal/GPU deployment, revision pinning for publication (`revision: null` → pinned sha in a `*_pinned.yaml`), vLLM-on-Linux verification. These are Phase-2 items — list them, don't start them.

## 4. Commands the agent should know

```bash
conda activate torch
pytest -q                       # offline suite (no downloads)
pytest -q -m model              # real SmolLM2 smoke (downloads ~360M)
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --dry-run
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --limit 5
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m.yaml --limit 20   # larger, still cheap
python scripts/inspect_run.py <run_dir> [<run_dir_B>]   # single + paired cc/cw/wc/ww
```

Key files: `scripts/evaluate.py`, `src/eval_harness/core/{interfaces,evaluator,config,results}.py`,
`src/eval_harness/{models/hf_causal_lm,tasks/gsm8k,parsing/gsm8k,scoring/gsm8k,prompting/chat,runtime/*}.py`,
`configs/gsm8k_smollm2_360m{,_debug,_vllm}.yaml`, `docs/evaluation_protocol.md` (frozen contract — any change to
prompt/parse/score/dataset interpretation requires a version bump + a note there).

## 5. Phase 2 progress vs `research_plan_phase2.md` (committed `9fe22ca`)

> Verdict: **scaffolding is in place, the intervention itself is not.**
> Configs validate, protocol wording is frozen, execution path exists —
> but no `RecirculationModelAdapter` exists yet, so Objectives B/C have
> not started. Do NOT run safety experiments, adaptive recirculation, or
> pass@128 yet (P2 §58).

### Objectives, models, configs (P2 §2–§4, §12–§13, §35, §50–§51)

| Plan | Status | Evidence / note |
|---|---|---|
| §2 Objective A trusted baseline | ⏳ Not started | Needs Gemma-1B-PT n=5 smoke on GPU |
| §2 Objectives B/C impl + reproduction | ⏳ Not started | Blocked on adapter |
| §3 Stage 2A (1B first), 2B (4B), 2C (12B optional) | ✅ Scaffolded | 1B configs exist (hf+vllm × baseline/recirc); 4B/12B = same layout, different model+L/S + pair (4B sha verified `cc012e0a…`) |
| §4 PT not IT (`family/size/tuning`) | ⚠️ Partial | Configs use PT checkpoints + `use_chat_template: false`; explicit `family/size/tuning` fields not yet in `ModelConfig` — add with adapter work or accept model-name-implies-it |
| §12–13 1B `11→4 α.15 β.85`, 4B `18→9 α.15 β1` dest-L2 | ✅ Configured | `gsm8k_gemma3_1b_pt_recirc{,_vllm}.yaml`; 4B twin still to write (trivial once 1B runs) |
| §35 freeze `docs/phase2_gsm8k_protocol.md` | ⚠️ Partial | PT/Kojima/BOS/indexing frozen in `evaluation_protocol.md` addendum; dedicated phase-2 file not yet split out |
| §50 manifest recirc fields | ✅ Ready | `InterventionConfig.to_dict()` records type/layers/alpha/beta/effective_beta/mixture/normalization/ramping |
| §51 run naming (`..._baseline`, `..._recirc_11_4_a015`) | ✅ Done | Experiment names follow the convention |

### Architecture & intervention abstraction (P2 §5–§11)

| Plan | Status | Evidence / note |
|---|---|---|
| §5 evaluator untouched, logic in model layer | ✅ By construction | No task/evaluator/scorer/parser changes this phase |
| §6 `Intervention` abstraction | ⚠️ Half-done | **Config side done** (typed + validated). Runtime side (adapter/intervention classes) not started |
| §7 baseline as `type: none` | ✅ Done | First-class, tested |
| §8 typed recirc config | ✅ Done | Incl. nested `mixture`/`normalization`/`ramping` shapes |
| §9 fixed training-free only | ✅ Scoped | No adaptive/MLP/finetune code exists |
| §10 destination-L2 norm + eps + norm logging | ⏳ Adapter work | Formula + debug-logging requirement captured; implement in adapter |
| §11 convex + nonconvex beta | ✅ Config-ready | `effective_beta` tested (0.85 / 1.0); adapter must honor it, never hardcode |

### Protocol & pass@1 (P2 §14–§17, §36–§37)

| Plan | Status | Evidence / note |
|---|---|---|
| §14 R1/R2/R3 separation | ✅ Documented | Kojima template comment + protocol addendum label ours "qualitative replication" |
| §15–16 zero-shot CoT, greedy pass@1, `max_new_tokens` 512 | ✅ Frozen | `gsm8k_kojima_v1` v1.0 + deterministic generation; same config for baseline/recirc |
| §17 no tuning on test | ✅ Policy | Paper params used verbatim; ramp=10 marked provisional |
| §36 lm-harness cross-check | ⏳ Scheduled | Required before any research claim; after first full runs |
| §37 reproduction classification honesty | ✅ Policy | Single-stage ≠ Kojima two-stage; documented, not hidden |

### Runs, comparison, analysis (P2 §18–§22, §52–§54)

| Plan | Status | Evidence / note |
|---|---|---|
| §18 Runs 1–6 (5 → 50 → full; 1B then 4B) | ⏳ Blocked | Needs Modal access (user) + adapter; `modal_app.py` scaffold ready, `py_compile` clean, never run |
| §19 paired transitions, §21 Δ-accuracy | ⚠️ Partial | `inspect_run.py` prints 4-cell table; file-output `compare_runs.py` (§20) not built |
| §20 `compare_runs.py` → comparison.json/csv | ⏳ Next scaffolding | Build before first paired runs |
| §22 generation-length analysis | ⏳ Analysis work | Lengths derivable from `raw_output` today; token counts TBD (store vs post-hoc) |
| §52–54 artifacts, figures, `analysis/gsm8k_phase2.py` | ⏳ Not started | After first real paired runs |

### Correctness mechanics (P2 §23–§31)

| Plan | Status | Evidence / note |
|---|---|---|
| §23–25 alpha=0 ≡ baseline; `src==dst` rejected; none≡recirc(α=0) test | ⚠️ Partial | Rejection enforced in config + tested; equivalence tests await the adapter |
| §26 cross-step recurrence (t→t+1, not same-step) | ⏳ Adapter core | Must be documented in code; vLLM RFC corroborates design (mix at `d`, rerun `d+1..N`, overwrite upper KV) |
| §27 not naive `d += α·s` | ⏳ Adapter core | Norm + beta + offset + state + boundary all required |
| §28 opt-in activation debug (norms, cosine; 1 ex, few positions) | ⏳ Adapter work | Config `ramping`/debug shape anticipated; implement with adapter |
| §29 0-based block indexing | ✅ Documented | Protocol addendum; adapter must assert `num_layers` bounds at init |
| §30 KV-cache semantics, §31 prefill vs decode | ⏳ Adapter + §32 hooks | Correctness-first; no premature optimization |

### Robustness, perf, acceptance (P2 §32–§34, §38–§40, §55–§62)

| Plan | Status | Evidence / note |
|---|---|---|
| §32 timing/memory instrumentation | ⏳ Evaluator hooks | wall/prefill/decode, tok/s, peak-mem (null on MPS); build before Runs 1–3 |
| §33 batch invariance 1/2/4 on recirc | ⏳ Awaits adapter | Harness proven on baseline; adapter must preserve it |
| §34 repeated-run determinism | ⏳ Awaits adapter | Same; rerun-compare procedure exists from Phase 1 |
| §38–40 pass@128 (separate mode + `pass_at_*` layout) | ⏳ Explicitly later | Only after pass@1 validated |
| §55 acceptance (suite green, 1B+4B load, full comparisons, identical reruns, paired compare, immutable runs) | ⏳ 1/11 | Suite green (71/1); rest pending |
| §56 report, §58 non-goals, §59 Phase-3 reuse, §60 commands, §61–62 done criteria | ⏳ Pending / scoped | `modal_app.py` covers §60 shape; no safety/adaptive code present (§58 clean) |

### Next steps (ordered)

- [x] **6. GPU runtime optimization (2026-09-22, local-verified, GPU-pending).**
  GPU was ~30% utilized because the recirc loop was serial prefill +
  batch-1 + 512-token caps on both stages + eager attention. Fixed:
  (A) per-stage budgets (`extraction_max_new_tokens`, 256/32 in Gemma
  configs — stage-2 answers need a handful of tokens, not 512);
  (B) SDPA on the GPU recirc config (eager kept for MPS only);
  (C) true inner-batched loop (left-pad, pad-gated hooks/state,
  per-row ramping/validity/finish, fixed-order sampling) + recirc
  batch_size 1→16. Proven locally: alpha=0 STILL bitwise-identical to
  HF baseline with padding in play, batch invariance now exercises real
  batching, determinism holds. Expected ~10x (40s/ex → ~4s/ex); GPU
  timing unconfirmed — validate on the next smoke before trusting it.

- [x] **0. Kojima two-stage prompting (2026-09-22).** Real Kojima
  protocol: stage 1 reasoning (`gsm8k_kojima_v1`), stage 2
  `"[X'] [Z] Therefore, the answer (arabic numerals) is"`
  (`gsm8k_answer_extract_v1` v1.0) through the SAME adapter +
  generation config; parser scores the final output. Evaluator
  auto-detects via optional `Task.build_extraction_prompt`
  (all-or-nothing, else loud error); records gain
  `reasoning`/`extraction_prompt`; manifest gains `prompt.extraction`
  (null unless extraction actually ran); `evaluation_code_version`
  bumped to `0.2.0` in all 10 configs. Proven: mock two-stage +
  batch invariance + real SmolLM2 smoke. Full suite: 109 passed,
  1 skipped. Previous single-stage GPU numbers are now superseded —
  re-run baselines under two-stage before any claim.

- [x] **1. `RecirculationModelAdapter` (2026-09-22).** Built in
  `src/eval_harness/models/recirculation.py`: subclasses
  `HFCausalLMAdapter` (reuses loading/device/dtype/tokenizer/revision),
  serial token-by-token cross-step loop, pre-hook at block `d+1` input,
  capture hook at block `s` output, warm-up at position 0, provisional
  linear ramp, opt-in `debug_steps` norm/cosine tracing (capped),
  `layer_indexing_convention` + `recurrence_variant` in metadata,
  layer bounds fail-fast via Hub config probe (backstop post-load).
  `scripts/evaluate.py` dispatches `intervention.type=recirculation`
  (hf) and rejects vLLM+recirc with an explanatory error.
- [x] **2. Invariant + determinism tests (2026-09-22).**
  `tests/unit/test_recirculation.py` (9 offline: mix math, eps safety,
  ramp, round-trip) + `tests/integration/test_recirculation.py`
  (`-m model`, SmolLM2 local): **alpha=0 outputs byte-identical to
  `HFCausalLMAdapter`**, twice-identical determinism, batch invariance,
  debug capture, bounds rejection. Full suite: 86 passed, 1 skipped.
- [x] **3. `compare_runs.py` + §32 timing hooks (2026-09-22).**
  `BatchTiming` convention (`interfaces.py`): adapters publish per-call
  input/output token lists + prefill/decode seconds; evaluator consumes
  per batch, aggregates, nulls anything unmeasured. Manifest gains a
  `performance` block (wall/prefill/decode, in/out tokens, tok/s, CUDA
  peak else null, coverage flag); `PredictionRecord` gains
  `input_tokens/output_tokens/question` (additive; old runs compare).
  HF reports totals, recirc a genuine prefill/decode split, vLLM
  best-effort, mock nulls. `scripts/compare_runs.py` writes
  comparison.json/csv + summary.json + report.txt into `comparisons/`
  (runs stay immutable): cells, deltas (relative null-safe),
  rescued/regressed, length stats + answer-offset heuristic.
  Proven live: old batch1-vs-batch2 identical (0 changed, coverage 0.00
  on pre-instrumentation runs); fresh HF n=2 (tok/s 13.0, coverage 1.00)
  vs fresh SmolLM2-recirc n=2 (prefill 12.9s / decode 9.2s — the paper's
  §31 serial-prefill phenomenon, measured). Full suite: 94 passed,
  1 skipped.
- [x] **4a. Modal access + GPU smoke Run 1 (2026-09-22).**
  vLLM engine fixed twice (spawn for fork bug, `FLASHINFER_SAMPLER=0`
  for missing nvcc) + `max_model_len: 4096` warmup trim. Gemma-3-1b-PT
  baseline n=5 on A100: **success** (`run_20260921_195252_1e08cf`,
  Kojima prompt, pins `fcf18a2a`/`740312ad`, greedy, batch 16,
  **1322 tok/s**, coverage 1.00, wandb `recirc-gsm8k` logged).
  Result 0/5 accuracy is a smoke-test non-claim (1B PT zero-shot).
  Known gaps: `peak_memory_bytes` was 0 (parent-process counter
  meaningless for vLLM → now recorded null), `git` null on Modal
  (no repo in image — bake a sha later if needed).
- [x] **4b. Recirc n=5 + first GPU paired compare (2026-09-22).**
  Recirc (`11→4 α.15`, HF backend on A100) n=5: success
  (`run_20260921_195803_b59052`, prefill 28.2s / decode 174.1s serial,
  peak 2.08 GB real, wandb logged). Paired vs vLLM baseline:
  5 common, 0.0 vs 0.0 acc (non-claim, 1B PT zero-shot), **2 changed
  examples**, shorter recirc traces (1348 vs 1564 mean chars);
  comparison + deltas logged to wandb (`comparisons/gpu_smoke_base_vs_recirc`).
  Caveat: backends differ (vLLM baseline vs HF recirc) — fine for smoke,
  full runs should share a backend where possible.
- [x] **4c. 50-pair on GPU, single-stage era (2026-09-22, superseded).**
  Baseline vs recirc: 50 common, 0.0 vs 0.0 acc, **39/50 changed**.
  (`comparisons/gpu_50_base_vs_recirc`.) Superseded by two-stage below.
- [x] **7. First two-stage GPU pair at n=50 (2026-09-22).**
  Optimized stack on CUDA (batched loop + SDPA + 256/32 budgets):
  recirc 194s/50ex (**3.9s/ex, ~10x faster** than the 40s/ex serial
  era), baseline vLLM 5.4s total. Result: 0.04 vs 0.04 acc (2 rescued
  + 2 regressed — flips both ways, no net effect at 1B), 47/50 outputs
  changed, short extraction outputs as designed. Logged to wandb
  (`comparisons/gpu_50_twostage_base_vs_recirc`).
- [x] **8. 4B scaling check at n=50 (2026-09-22).**
  First attempt failed fast: Gemma3 text checkpoints load as
  `Gemma3ForCausalLM` whose `.model` is the multimodal `Gemma3Model`
  wrapper (no `.layers`) — fixed with a layout resolver
  (`model.layers` → `model.language_model.layers` → `transformer.h`)
  + robust hidden-size lookup, covered by offline layout tests.
  Reran clean. Result: baseline 0.24 (12/50) vs recirc 0.26 (13/50),
  delta +0.02 — **9 rescued, 8 regressed, 39/50 changed**. Net +1 is
  noise at n=50, but the churn pattern + positive sign is the first
  directionally-paper-consistent signal (1B showed 0.04/0.04).
  Recirc cost 255s/50ex (~5s/ex), peak 9.7 GB. Logged to wandb
  (`comparisons/gpu_4b_twostage_base_vs_recirc`). Full 1B/4B pairs
  still required before any claim.
- [ ] **6. Defer explicitly:** 4B/12B configs (trivial clones once 1B
  runs), sweeps (§41–48), stats (§49), pass@128 (§38–40), report (§56).
