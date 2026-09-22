# Phase 2 GSM8K protocol — FROZEN after the 1B/4B full runs (2026-09-22)

> DO NOT MODIFY. Any change requires a new protocol version + new runs.
> Source of truth for every value below: the run manifests
> (`manifest.json` → `prompt`/`generation`/`task`/`model` blocks).

## Models (both pinned; weights dated 2025-03-21, pre-paper)

| | 1B | 4B |
|---|---|---|
| repo | `google/gemma-3-1b-pt` | `google/gemma-3-4b-pt` |
| model revision | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` |
| tokenizer revision | same as model | same as model |
| tuning | pretrained (base) | pretrained (base) |
| dtype | bf16 (HF) / auto (vLLM) | bf16 (HF) / auto (vLLM) |

## Dataset

- `openai/gsm8k`, config `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`
- split `test` (1319 examples), order = raw dataset order, stable ids
  `gsm8k-test-{index:05d}-{sha8(question)}`

## Prompting (two-stage Kojima)

- Stage 1 `gsm8k_kojima_v1` v1.0, hash
  `8fc86acdb3e36fa05d7951a22392e8759e94c4351a58335715e1ffaff77fa7ae`:
  `Q: {question} A: Let's think step by step.`
- Stage 2 `gsm8k_answer_extract_v1` v1.0, hash
  `1801dee66c805340014eac7d7f0bfb47ca5f7abc599c5222b64d4172be971701`:
  input = `[stage-1 prompt] [stage-1 reasoning] Therefore, the answer
  (arabic numerals) is`
- Chat template: OFF (`use_chat_template: false`, raw prompt for PT).
- BOS: required at window start, adapter-guarded (paper-v2 confound).

## Generation (identical for baseline and treatment)

`do_sample=false, temperature=0.0, top_p=1.0, seed=42`,
`max_new_tokens=256` (reasoning), `extraction_max_new_tokens=32`.

## Parsing / scoring

- `gsm8k_parser v1.0` (after-last-`####` → `\boxed` → last number;
  reference = after-last-`####`), scored on the STAGE-2 output.
- `gsm8k_exact_match v1.0` (normalized exact match + float fallback;
  parse failures score 0 and stay in the denominator).

## Interventions (0-based transformer blocks)

| | 1B | 4B |
|---|---|---|
| source → destination | 11 → 4 | 18 → 9 |
| alpha / beta | 0.15 / 0.85 (convex) | 0.15 / 1.0 (nonconvex) |
| normalization | destination-L2, eps 1e-6 | destination-L2, eps 1e-6 |
| ramping | linear over first 10 tokens (provisional schedule) | off |
| recurrence | cross-step tokenwise serial: deep@t → shallow@t+1, warm-up at 0 | same |

## Versions

`task_version=1.0`, `evaluation_code_version=0.2.0`,
`result_schema_version=1.0`, `parser/scorer 1.0`.

## Official full runs (all limits null, greedy, seed 42)

| run | dir suffix | acc |
|---|---|---|
| 1B baseline (vLLM) | `run_20260921_213746_f0d8ab` | 0.0167 (22) |
| 1B recirc (HF) | `run_20260921_213557_173592` | 0.0144 (19) |
| 4B baseline (vLLM) | `run_20260921_214339_11eb2e` | 0.2942 (388) |
| 4B recirc (HF) | `run_20260921_213705_b30f77` | 0.2775 (366) |

## Known protocol deltas vs the paper (see progress.md §9)

Single most important: our one-pass-per-step cross-step loop vs the
paper's two-stack unrolling (Fig 3c) — different recirculation
schedules that must be disentangled before any reproduction claim.
Also: paper doesn't publish model shas / max-tokens / answer-parsing
code; our choices are documented above instead of guessed.
