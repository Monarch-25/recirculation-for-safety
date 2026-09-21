# LLM Evaluation Harness

Research-grade, modular harness for GSM8K baseline evaluation. Model-agnostic
evaluator + swappable model adapters (baseline today, Recirculation later).

## Install (conda env `torch`)

```bash
conda activate torch
pip install -e ".[dev]"   # or: pip install -e . && pip install pytest
```

## Run tests (offline, no downloads)

```bash
conda activate torch
pytest -q
```

Model-backed smoke test (downloads SmolLM2-360M, Apple Silicon OK):

```bash
pytest -q -m model
```

## Dry run (no generation)

```bash
conda activate torch
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --dry-run
```

## Smoke test (5 real examples)

```bash
conda activate torch
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_debug.yaml --limit 5
```

## Larger run

```bash
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m.yaml --limit 20
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m.yaml   # full 1319
```

CLI overrides: `--limit N`, `--batch-size N`.

## Inference backends

`model.backend` selects the inference adapter; everything else
(task, prompts, parsing, scoring, artifacts) is unchanged:

- `hf` (default): plain transformers, works on CPU/MPS/CUDA.
- `vllm`: optimized inference on Linux/CUDA (`pip install -e ".[vllm]"`).
  vLLM publishes Linux-x86_64 wheels only — it cannot run on Apple
  Silicon macOS, where the offline suite skips its tests.

```bash
# Linux/GPU example (same GSM8K protocol, larger batches)
python scripts/evaluate.py --config configs/gsm8k_smollm2_360m_vllm.yaml --limit 20
```

## Inspect runs

```bash
python scripts/inspect_run.py results/gsm8k/smollm2-360m-instruct/run_XXX
python scripts/inspect_run.py results/.../run_A results/.../run_B   # paired A->B table
```

## Results layout

```text
results/gsm8k/smollm2-360m-instruct/run_YYYYMMDD_HHMMSS_<id>/
  manifest.json  config.yaml  metrics.json  predictions.jsonl
  environment.json  logs.txt
```

## Design

`Task` owns benchmark semantics; `ModelAdapter` owns generation;
`Evaluator.evaluate(task, model, config)` orchestrates. Swapping in a
`RecirculationModelAdapter` later changes nothing else. See
`docs/evaluation_protocol.md` for the frozen benchmark contract.
