# Deep-to-Shallow Recirculation for LLM Safety

> **Research project:** Can inference-time deep-to-shallow representation recirculation improve LLM safety without changing model weights, while preserving reasoning and benign utility?

**Status:** 🟢 Phase 2 core complete — full GSM8K pairs, sweep, handover written
**Current stage:** Phase 2 reporting closed; Phase 3 safety spec exists (`docs/research_plan_phase3.md`), prompts pending.  
**Primary evaluation:** GSM8K → safety benchmarks → representation/mechanistic analysis  
**Target runtime:** Apple Silicon for development → CUDA/Modal for larger experiments

---

## 1. Project in one minute

This repository investigates a simple hypothesis:

> Information that becomes available at deeper transformer layers may contain useful contextual or safety-relevant state. Feeding that representation back into a shallower layer at the next inference step may make that information available earlier in subsequent computation.

The project combines two recent directions:

1. **Recirculation** — an inference-time architectural modification that introduces a specific form of recurrence by feeding deeper representations back into shallower computation across successive input steps. The current paper reports improvements in perplexity, GSM8K, and other downstream tasks while keeping model weights frozen. [Mozer et al., *Recirculation*, arXiv:2608.17981](https://arxiv.org/abs/2608.17981)
2. **ALIGNBEAM** — an inference-time safety/alignment method that transfers safety preferences through cross-vocabulary logit mixing and evaluates harmful refusal together with benign utility. [Chawla et al., *ALIGNBEAM*, arXiv:2606.12342](https://arxiv.org/abs/2606.12342)

The long-term question is therefore not simply:

> "Does Recirculation make a model refuse more?"

Instead:

> **Does deep-to-shallow representation transfer change the model's safety behavior in a useful way, without turning into blanket refusal or sacrificing reasoning capability?**

This repository is being built so that the answer is supported by reproducible experiments rather than one-off benchmark numbers.

---

## 2. Research questions

### RQ1 — Capability

Does Recirculation reproduce the reported improvement in language-model evaluation and reasoning, particularly GSM8K?

### RQ2 — Safety

Does the same inference-time representation intervention change safety behavior on harmful prompts?

### RQ3 — Calibration

If harmful refusal increases, does benign over-refusal also increase?

### RQ4 — Depth

Are safety effects localized to particular source/destination layer pairs and mixture strengths?

### RQ5 — Mechanism

If a safety effect exists, is it associated with safety-relevant information emerging more strongly at deeper layers?

### RQ6 — Intervention space

How does representation-space intervention compare with output/logit-space alignment methods such as ALIGNBEAM?

---

## 3. Central hypothesis

For a deeper source layer \(s\), shallower destination layer \(d\), and step \(t\), the core intervention is conceptually:

\[
h_d^{t+1}
\leftarrow
\beta h_d^{t+1}
+
\alpha\, f(h_s^t)
\]

where \(f(\cdot)\) performs the required normalization/scaling.

The important detail is that the connection is **cross-depth and cross-step**:

```text
step t
  deep layer
      │
      │ source representation
      ▼
  stored recurrent state
      │
      ▼
step t+1
  shallow layer
```

This is different from simply adding a same-step skip connection or looping through a stack of layers.

The first scientific claim we are testing is behavioral:

\[
\Delta \text{harmful safety} > 0
\]

while maintaining approximately stable:

\[
\Delta \text{benign over-refusal}
\]

and useful task performance.

A stronger mechanistic claim will only be made if later causal representation experiments support it.

---

## 4. Research status

### Current repository snapshot

The repository currently contains the research evaluation foundation:

- Modular `Task` / `ModelAdapter` / `Evaluator` separation.
- GSM8K evaluation pipeline.
- Hugging Face inference backend.
- vLLM backend scaffold for Linux/CUDA execution.
- Dry-run and smoke-test workflow.
- Batch-size CLI configuration.
- Per-run artifacts including manifests, metrics, predictions, and environment metadata.
- Tests designed to run without downloading a model.
- Modal runtime scaffolding for later GPU execution.

The repository homepage currently exposes this structure under:

```text
configs/
docs/
results/
scripts/
src/eval_harness/
tests/
modal_app.py
pyproject.toml
```

The current evaluation README documents a 360M-parameter SmolLM2 development path and the same harness architecture intended for future Recirculation adapters.

---

## 5. Research roadmap

The project is organized into gated phases.

```text
Phase 1
Evaluation infrastructure
        │
        ▼
Phase 2
Trusted GSM8K + Recirculation reproduction
        │
        ▼
Phase 3
Safety behavior evaluation
        │
        ▼
Phase 4
Representation / causal mechanism
        │
        ▼
Phase 5
ALIGNBEAM comparison + combined methods
        │
        ▼
Phase 6
Robustness, scaling, systems analysis
        │
        ▼
Paper / release
```

A phase is not considered complete merely because code exists; its acceptance criteria are defined in `docs/RESEARCH_PROGRESS.md`.

---

## 6. Repository architecture

The project intentionally keeps benchmark semantics independent from model implementation.

```text
                    Evaluation Engine
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
          GSM8K       Safety Tasks     Future Tasks
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                      ModelAdapter
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          HF model     Recirculation   Future methods
```

The core execution path is:

```text
Dataset
  ↓
Task
  ↓
Prompt construction
  ↓
ModelAdapter
  ↓
Generation
  ↓
Parser / evaluator
  ↓
Metrics
  ↓
Per-example results
  ↓
Run manifest + report
```

This separation is deliberate.

A future Recirculation implementation should be able to replace the baseline inference adapter without changing:

- the GSM8K task,
- the prompts,
- the scorer,
- the result schema,
- the reproducibility metadata,
- or the analysis code.

---

## 7. Current repository layout

```text
recirculation-for-safety/
│
├── configs/
│   └── experiment configurations
│
├── docs/
│   ├── evaluation_protocol.md
│   └── RESEARCH_PROGRESS.md
│
├── results/
│   └── immutable experiment artifacts
│
├── scripts/
│   ├── evaluate.py
│   └── inspect_run.py
│
├── src/
│   └── eval_harness/
│       ├── core/
│       ├── models/
│       ├── tasks/
│       ├── prompting/
│       ├── parsing/
│       ├── scoring/
│       ├── runtime/
│       └── utils/
│
├── tests/
│
├── modal_app.py
├── pyproject.toml
└── README.md
```

The exact implementation may evolve. The separation of concerns should not.

---

## 8. Getting started

### Development environment

The initial development path is designed for an Apple Silicon Mac.

```bash
conda activate torch
pip install -e ".[dev]"
```

Run the offline tests:

```bash
pytest -q
```

Run the dry-run:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml \
  --dry-run
```

Run the five-example smoke test:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml \
  --limit 5
```

Run a larger development evaluation:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m.yaml \
  --limit 20
```

The existing harness also supports a CUDA/Linux vLLM path for later GPU experiments.

---

## 9. Reproducibility is part of the experiment

Every research run should be traceable to the exact configuration that produced it.

The harness is designed to capture at least:

| Provenance | Captured |
|---|---|
| Git commit | ✅ |
| Git branch / dirty state | ✅ |
| Python version | ✅ |
| PyTorch version | ✅ |
| Transformers version | ✅ |
| CUDA version | ✅ when available |
| GPU model | ✅ when available |
| Model revision | ✅ |
| Dataset revision | ✅ |
| Tokenizer revision | ✅ |
| dtype | ✅ |
| seed | ✅ |
| generation config | ✅ |
| batch size | ✅ |
| prompt template | ✅ |
| task version | ✅ |
| evaluation code version | ✅ |

A run should also retain:

```text
manifest.json
config.yaml
metrics.json
predictions.jsonl
environment.json
logs.txt
```

The objective is that a reported number such as:

```text
GSM8K = 61.2%
```

is never detached from the protocol that produced it.

---

## 10. Research data flow

For benchmark research, aggregate metrics are not enough.

Each example should remain traceable:

```text
example_id
    ↓
prompt
    ↓
model output
    ↓
parsed answer / safety judgment
    ↓
correct / safe / unsafe
```

This enables paired analyses such as:

```text
baseline correct  → Recirculation correct
baseline correct  → Recirculation wrong
baseline wrong    → Recirculation correct
baseline wrong    → Recirculation wrong
```

For safety experiments the same pattern becomes:

```text
baseline unsafe → Recirculation safe
baseline safe   → Recirculation unsafe
```

and, for benign inputs:

```text
baseline answer    → Recirculation refusal
baseline refusal   → Recirculation answer
```

This is important because an improvement in an aggregate refusal rate can conceal a simultaneous increase in benign over-refusal.

---

## 11. Phase 1 — Evaluation infrastructure

**Goal:** build a trustworthy evaluation substrate before making a research claim.

Current status:

- [x] Model-independent evaluator architecture
- [x] Task abstraction
- [x] Hugging Face adapter
- [x] GSM8K task
- [x] Prompting abstraction
- [x] Parsing/scoring separation
- [x] Config-driven execution
- [x] Dry-run
- [x] Small smoke test
- [x] Per-example result storage
- [x] Reproducibility metadata
- [x] Run inspection
- [x] Offline tests
- [x] Modal/CUDA execution path scaffold

See `docs/RESEARCH_PROGRESS.md` for the detailed acceptance criteria.

---

## 12. Phase 2 — Trusted GSM8K + Recirculation reproduction

The first real research phase is capability, not safety.

The sequence is:

```text
small-model validation
        ↓
research-model baseline
        ↓
Recirculation implementation
        ↓
implementation invariants
        ↓
GSM8K reproduction
        ↓
development layer/alpha analysis
```

Primary research models:

- Gemma 3 1B PT
- Gemma 3 4B PT
- optionally Gemma 3 12B PT

The current Recirculation paper reports selected source/destination pairs of:

```text
1B  : 11 → 4
4B  : 18 → 9
12B : 35 → 16
```

and reports downstream capability improvements including GSM8K gains. See the paper for the exact experimental protocol and model/version details.

Phase 2 should first reproduce the externally specified configuration before using any GSM8K development subset for our own layer/alpha search.

---

## 13. Phase 3 — Safety evaluation

Once the capability pipeline is trusted, evaluate:

### Harmful/adversarial behavior

- HarmBench-Standard
- HarmBench-Contextual
- AdvBench
- Sorry-Bench
- WildJailbreak

### Benign / over-refusal behavior

- XSTest
- OR-Bench-Hard
- JBB-Benign

These benchmarks are motivated in part by the evaluation protocol used in recent inference-time alignment research, including ALIGNBEAM.

The safety analysis will report separate dimensions rather than hiding everything inside a single "safety score":

```text
harmful refusal / safety judgment
benign over-refusal
benign utility
GSM8K capability
latency / memory
```

---

## 14. Phase 4 — Mechanistic investigation

If a robust behavioral effect is found, the project will investigate why.

Planned analyses include:

```text
layer-wise safety probes
activation norms
cross-layer similarity
harmful vs benign representation trajectories
source/destination intervention analysis
activation ablation
representation perturbation
```

The goal is to test the stronger hypothesis:

> Safety-relevant information becomes more explicit at deeper layers and Recirculation makes part of this information available to shallower computation.

Behavioral correlation alone will not be treated as proof of this mechanism.

---

## 15. Phase 5 — ALIGNBEAM comparison

ALIGNBEAM operates in logit/output space.

This project operates in representation space.

The eventual comparison is therefore:

```text
Representation space
        ↓
Deep representation
        ↓
Recirculation
        ↓
Shallow state
        ↓
Generation
```

versus:

```text
Output space
        ↓
Safe anchor logits
        ↓
Cross-vocabulary translation / mixing
        ↓
Candidate generation / selection
```

The eventual experiment will compare:

```text
Baseline
Recirculation
ALIGNBEAM
Recirculation + ALIGNBEAM
```

The combined condition is particularly useful because it can test whether representation-space and output-space interventions have additive, independent, or interacting effects.

This phase comes only after the standalone methods are independently validated.

---

## 16. What would constitute a meaningful result?

The strongest version of the hypothesis would produce a pattern such as:

```text
harmful safety       ↑
benign over-refusal  ≈
GSM8K                 ≈ / ↑
```

together with:

```text
specific depth dependence
+
compute-matched controls
+
robustness across datasets
+
representation-level evidence
```

But the project is explicitly designed to accommodate other outcomes.

For example:

```text
GSM8K ↑, safety unchanged
```

would suggest a capability/contextualization effect without clear safety benefits.

Likewise:

```text
GSM8K ↑, harmful refusal ↓
```

would suggest a potentially important capability/alignment trade-off.

A null or negative result is therefore a valid research outcome.

---

## 17. Experimental discipline

The following rules are part of the research protocol.

### Freeze before testing

Do not repeatedly tune:

```text
layer pair
alpha
beta
prompt
decoding
```

on the final test set.

Use:

```text
development
→ validation
→ frozen final test
```

instead.

### Preserve raw outputs

Do not throw away individual generations after computing an aggregate score.

### Compare paired examples

Whenever baseline and treatment see the same benchmark item, retain the example ID and analyze the transition.

### Report denominators

Every percentage should be accompanied by `N` / valid `N`.

### Separate evidence from interpretation

For example:

```text
Observed:
Recirculation increased refusal on dataset X.

Hypothesis:
Deeper safety-relevant representation may be responsible.
```

The second statement should not be presented as established until mechanistic evidence supports it.

---

## 18. Relation to established research repositories

This repository intentionally follows conventions that are useful in mature open research codebases:

- **Unified evaluation interfaces and explicit task/prompt configuration**, as exemplified by EleutherAI's `lm-evaluation-harness`. It emphasizes standardized tasks, reproducible prompts, configurable model backends, and a clear evaluation API. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Explicit safety evaluation documentation and saved individual evaluation results**, as seen in AllenAI's Open-Instruct evaluation stack. [Open-Instruct safety evaluations](https://github.com/allenai/open-instruct/blob/main/docs/safety.md)
- **Reproduction-oriented documentation and commands**, including configuration and environment details, as used in mature research codebases such as Open-Instruct. [Open-Instruct](https://github.com/allenai/open-instruct)
- **Explicit experiment logging and iterative research artifacts**, a useful convention highlighted by Karpathy's `autoresearch`. [autoresearch](https://github.com/karpathy/autoresearch)

The goal here is not to reproduce any of those projects' architectures. It is to adopt the research-engineering habits that make experiments understandable to someone other than the original author.

---

## 19. Research artifacts

As experiments mature, this repository should accumulate:

```text
configs/
    frozen experiment configurations

results/
    immutable run artifacts

docs/
    evaluation protocols
    research progress
    experiment notes

figures/
    generated figures only

reports/
    benchmark and experiment reports
```

Do not manually edit benchmark numbers into figures or reports.

Prefer generating them from the stored result artifacts.

---

## 20. Reproducing a result

The intended workflow is:

```text
1. Checkout the recorded Git commit.
2. Install the documented environment.
3. Load the experiment config.
4. Verify model/dataset/tokenizer revisions.
5. Run the exact command stored in the manifest.
6. Compare generated artifacts.
```

A researcher should not need private notebook state to understand how a headline result was produced.

---

## 21. Project conventions

### Experimental code

Small, composable modules are preferred over benchmark-specific scripts.

### Configuration

Experiment parameters belong in versioned config files.

### Results

Do not overwrite previous runs.

### Tests

Evaluation infrastructure should be tested without requiring expensive model execution.

### Notebooks

Notebooks may be used for exploration and visualization, but official numbers should be generated from scripts/configurations that can be rerun.

### External dependencies

Pin or record versions whenever they can affect a benchmark result.

---

## 22. Current limitations

This repository is a research prototype.

At the current stage:

- the safety benchmark suite is not yet the primary validated path;
- the mechanistic analysis pipeline is not yet established;
- large-scale GPU experiments are not yet the default development workflow;
- final benchmark conclusions have not been established;
- exact reproduction of published numbers should not be assumed until the model revisions, prompts, decoding, dataset versions, and evaluator protocol are matched.

The progress tracker records the actual state of these items.

---

## 23. Contributing / research collaboration

For research changes, prefer small, reviewable commits.

A useful research PR should answer:

```text
What hypothesis or infrastructure requirement does this change address?
What experiment does it enable?
What invariant or test protects against regression?
What metadata will be recorded?
```

When introducing a new benchmark or intervention, preserve the separation:

```text
benchmark semantics ≠ inference implementation
```

---

## 24. Citation

If you use the Recirculation method or this repository in research, cite the underlying paper and identify the repository commit/configuration used for your experiments.

### Recirculation

Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer, Sunny Sanyal, Rosanne Liu.

*Recirculation.*

https://arxiv.org/abs/2608.17981

### ALIGNBEAM

Chirag Chawla, Pratinav Seth, Vinay Kumar Sankarapu.

*ALIGNBEAM: Inference-Time Alignment Transfer via Cross-Vocabulary Logit Mixing.*

https://arxiv.org/abs/2606.12342

Repository:

https://github.com/Monarch-25/recirculation-for-safety

---

## 25. Research status

**Last reviewed:** 2026-09-22

Current milestone:

> **Build a trusted evaluation substrate first; establish GSM8K capability baselines next; only then make safety claims.**

See:

[`docs/RESEARCH_PROGRESS.md`](docs/RESEARCH_PROGRESS.md)

for the living experiment checklist.
