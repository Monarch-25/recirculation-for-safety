# Evaluation protocol (benchmark contract) — v0.1.0

This document is the frozen contract for the GSM8K baseline. Any change to
the items below requires a task/prompt/parser version bump and a note here.

## Dataset

- **Dataset:** `openai/gsm8k`, config `main` (fallback: legacy `gsm8k`).
  Rationale: the canonical `gsm8k` repo was migrated to the `openai` org;
  both ids are tried so the harness keeps working.
- **Split:** `test` (1319 examples when `limit: null`).
- **Revision:** resolved Hub commit sha recorded as
  `manifest.task.dataset_revision`; user may pin `task.dataset_revision`.
- **Example IDs:** source rows have no stable id, so
  `example_id = gsm8k-test-{index:05d}-{sha8(question)}` where `sha8` is the
  first 8 hex chars of SHA-256 over the question text. Deterministic in
  dataset order; content hash catches silent dataset reordering.
- **Order:** raw dataset order, truncated to `limit` when set.

## Prompt template (`gsm8k_cot_v1`, version `1.0`)

Exact template (frozen):

```text
Solve the following math problem step by step.

{question}

Give the final answer clearly.
```

- Rendered deterministically via `str.format(question=...)`.
- Template name/version/hash recorded in `manifest.prompt`; full rendered
  prompt saved per example in `predictions.jsonl`.
- **Chat template:** for instruct models the adapter wraps the rendered
  prompt as a single `user` message via
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)` when the
  tokenizer defines a chat template (`model.use_chat_template: true`).
  With `false`, or no chat template, the raw prompt is used.

## Decoding

- Greedy deterministic: `do_sample=false, temperature=0.0, top_p=1.0`.
- `max_new_tokens=512` (debug config: 256), `seed=42`.
- Seeded via `random`, `numpy`, `torch.manual_seed` before generation.
- **Attention backend:** `model.attn_implementation=eager` is pinned in the
  provided configs because the SDPA fast path aborts (SIGABRT) on Apple
  Silicon MPS with torch 2.6 for SmolLM2-360M. Eager attention is
  mathematically identical, only slower; the setting is recorded in
  `manifest.model.extra`/metadata. Re-evaluate on CUDA/torch upgrades.
- MPS note: Apple MPS does not guarantee bit-for-bit reproducibility
  across OS/torch versions; same-version reruns are expected to be
  near-deterministic for greedy decoding but this is not contractually
  guaranteed. CUDA determinism additionally depends on cuDNN flags.

## Backend parity (hf vs vLLM)

- Both adapters implement `ModelAdapter.generate(prompts, config)` and
  share `prompting.chat.format_for_chat`, so prompts are byte-identical.
- `do_sample=false` → vLLM `temperature=0.0` (greedy); sampling configs
  pass through; `seed` is set on the engine and per request.
- vLLM-specific knobs (`tensor_parallel_size`, `gpu_memory_utilization`,
  `enforce_eager`, `max_model_len`, `vllm_version`) are recorded in
  `manifest.model`; `backend` is recorded alongside `device`.
- Cross-backend equality is expected for greedy decoding but must be
  verified empirically (tokenizer stop-handling can differ); do not assume
  it for research claims without a paired comparison run.

## Answer extraction (`gsm8k_parser v1.0`)

1. Reference: substring after last `####`, else last number in string.
2. Prediction: (a) after last `####`; (b) last `\boxed{...}`; (c) last
   number-like token (`-?\d[\d,]*\.?\d*`).
3. Normalization: strip whitespace/commas/`$`/surrounding punctuation.
4. No match → `parse_success=false`, `parsed_answer=null`.

## Scoring (`gsm8k_exact_match v1.0`)

- `correct = normalize(pred) == normalize(ref)` with numeric fallback
  (`float(a) == float(b)`, so `42` ≡ `42.0`).
- Parse failures score 0 and **remain in the denominator**:
  `accuracy = num_correct / num_examples`.

## Aggregation

`metrics.json`: `num_examples`, `num_correct`, `accuracy`,
`parse_rate = num_parsed / num_examples`, `num_parse_failures`.

## Result schema (`result_schema_version 1.0`)

Run dir `results/<task>/<model-slug>/run_<UTC-stamp>_<shortid>/` holds
`manifest.json`, `config.yaml`, `metrics.json`, `predictions.jsonl`,
`environment.json`, `logs.txt`. Never overwritten; each run is unique.

## Versions

- `task_version = 1.0`, `evaluation_code_version = 0.1.0`.
- Bump task version on any change to prompt/parsing/scoring/dataset
  interpretation; bump `evaluation_code_version` on evaluator/artifact
  changes; record git commit/branch/dirty always.

## Future cross-check

Before research claims, validate against `lm-evaluation-harness`
(`lm_eval --model hf --tasks gsm8k`) under the same greedy protocol and
document any delta (prompt wording and answer regex are the usual sources).

## Phase 2 addendum — PT models and Kojima wording

Applies to `google/gemma-3-{1b,4b,12b}-pt` runs. The v0.1.0 contract above
is unchanged for instruct models; this section freezes the PT variant.

- **Prompt template (`gsm8k_kojima_v1`, version `1.0`)**, paper-following
  wording (Kojima et al. 2022):
  ```text
  Q: {question} A: Let's think step by step.
  ```
  SINGLE-STAGE: one generation per example + the unchanged
  `gsm8k_parser v1.0` / `gsm8k_exact_match v1.0`. This is NOT Kojima's
  two-stage protocol (no second "Therefore, the answer (arabic numerals)
  is" extraction call). Comparisons against the paper are therefore
  **qualitative replications**, never exact reproductions.
- **PT rendering:** `model.use_chat_template: false`. Base models get the
  raw prompt with no chat-template wrapping.
- **BOS requirement (paper-v2 confound):** every input window MUST start
  with BOS. Adapters verify the first token id after tokenization and log
  a warning otherwise (`prompting.chat.check_bos_present`). Gemma results
  without BOS are not trustworthy.
- **Layer indexing:** 0-based transformer blocks; `source_layer >
  destination_layer` enforced at config load (identity rejected).
  Semantics: mix at boundary `d`, rerun layers `d+1..N`, upper KV
  overwritten (corroborated by the public vLLM Recirculation RFC).
- **Ramping (1B, provisional):** linear ramp of alpha over the first
  `ramp_tokens` (default 10, RFC-following). The exact paper schedule is
  TBD; whatever is used is recorded in `manifest.model.intervention`.
- **Pinned vs floating configs:** `configs/*_pinned.yaml` are official
  (pinned model/tokenizer/dataset revisions). Unpinned twins are for
  iteration. In both cases the manifest sha is the canonical record.
