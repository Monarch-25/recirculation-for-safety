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
