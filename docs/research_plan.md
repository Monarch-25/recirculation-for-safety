You are implementing the first version of a **research-grade, modular LLM evaluation harness** for an upcoming research project on **deep-to-shallow representation recirculation for improving LLM reasoning and safety**.

The immediate goal is NOT to implement Recirculation.

The immediate goal is to build and validate a **correct, reproducible, model-agnostic evaluation framework** that can:

1. Evaluate GSM8K.
2. Run a baseline Hugging Face causal LM.
3. Save every raw prediction and all metadata required for reproducibility.
4. Separate dataset loading, prompting, model invocation, answer parsing, scoring, aggregation, and experiment tracking.
5. Be testable locally on an Apple Silicon Mac without a GPU.
6. Later allow swapping the model adapter for a custom Recirculation implementation without changing the evaluator or benchmark code.
7. Eventually run unchanged inside a GPU-based Modal sandbox.

The code must be modular and research-oriented. Avoid building a quick single-file script.

---

# 1. Research context

The eventual research project will compare:

* standard baseline inference
* deep-to-shallow representation Recirculation
* potentially other inference-time interventions such as layer looping or logit-space interventions
* later, safety-specific interventions

The eventual research questions involve comparing models under exactly the same benchmark and prompting protocol.

Therefore the architecture must enforce this separation:

```text
Dataset
   ↓
Task
   ↓
Prompt construction
   ↓
Model Adapter
   ↓
Generation
   ↓
Prediction Parser
   ↓
Metric / Scorer
   ↓
Aggregated Evaluation Result
   ↓
Experiment Artifact Store
```

The evaluator MUST NOT contain model-specific logic.

The model adapter MUST NOT contain benchmark-specific logic.

The task MUST NOT know whether the model is baseline, Recirculation, or some future intervention.

---

# 2. Immediate scope

Implement only the following initially:

### Benchmark

GSM8K.

### Model

Use a small Hugging Face causal LM with fewer than 500M parameters.

Prefer:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

or another stable instruct causal model below 500M parameters if that model is unavailable.

Do NOT begin with Gemma 1B/4B/12B.

### Hardware

The first execution environment will be:

```text
Apple Silicon Mac M2
```

There may be no CUDA.

The harness must therefore gracefully represent CUDA as unavailable rather than assuming CUDA exists.

### Evaluation scale

Do NOT run the complete GSM8K test set as the initial functionality test.

Support a configurable number of examples, e.g.:

```text
--limit 5
```

or

```text
--limit 20
```

The first validation run should use only a small number of examples.

The framework itself, however, must support the full 1319-example GSM8K test set later without architectural changes.

---

# 3. Critical architectural principle

Do NOT write:

```python
evaluate(dataloader, model_invocation_method)
```

as the primary architecture.

Instead define explicit interfaces.

The benchmark should conceptually be invoked as:

```python
result = evaluator.evaluate(
    task=gsm8k_task,
    model=model_adapter,
    config=eval_config,
)
```

The evaluator owns the orchestration.

The model adapter owns generation.

The task owns:

* dataset loading
* prompt construction
* answer extraction
* task-specific scoring

The result system owns:

* metrics
* per-example records
* manifests
* reproducibility metadata

---

# 4. Required project structure

Create a clean repository structure similar to:

```text
llm-eval-harness/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── gsm8k_smollm2_360m.yaml
│   └── gsm8k_smollm2_360m_debug.yaml
│
├── src/
│   └── eval_harness/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── interfaces.py
│       │   ├── evaluator.py
│       │   ├── config.py
│       │   └── results.py
│       │
│       ├── models/
│       │   ├── base.py
│       │   └── hf_causal_lm.py
│       │
│       ├── tasks/
│       │   ├── base.py
│       │   └── gsm8k.py
│       │
│       ├── prompting/
│       │   └── chat.py
│       │
│       ├── parsing/
│       │   ├── base.py
│       │   └── gsm8k.py
│       │
│       ├── scoring/
│       │   ├── base.py
│       │   └── gsm8k.py
│       │
│       ├── runtime/
│       │   ├── environment.py
│       │   ├── hardware.py
│       │   └── reproducibility.py
│       │
│       └── utils/
│           ├── hashing.py
│           ├── io.py
│           └── logging.py
│
├── scripts/
│   ├── evaluate.py
│   └── inspect_run.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── results/
│
└── docs/
    └── evaluation_protocol.md
```

You may improve this structure if there is a strong reason, but maintain the same separation of concerns.

---

# 5. Core interfaces

Implement explicit interfaces/protocols.

## Model Adapter

Create a model abstraction approximately equivalent to:

```python
class ModelAdapter(ABC):

    @property
    def model_id(self) -> str:
        ...

    @property
    def model_revision(self) -> str | None:
        ...

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        ...
```

The evaluator must only depend on this abstraction.

Create:

```text
HFCausalLMAdapter
```

as the first implementation.

Later we should be able to add:

```text
RecirculationModelAdapter
VLLMModelAdapter
ModalModelAdapter
APIModelAdapter
```

without changing the evaluator.

---

# 6. ModelAdapter requirements

The Hugging Face adapter should support:

* causal language models
* CPU
* MPS
* CUDA
* automatic device detection
* configurable dtype
* batched generation
* deterministic generation when sampling is disabled
* tokenizer loading
* model revision pinning
* tokenizer revision pinning

Do not assume CUDA.

Device resolution should conceptually be:

```text
CUDA if explicitly requested and available
otherwise MPS if requested/available
otherwise CPU
```

But make this configurable rather than silently overriding user configuration.

The adapter must expose sufficient metadata for the run manifest.

---

# 7. Model revision handling

Do not rely only on:

```python
model_name = "..."
```

Record:

* model name
* model revision
* tokenizer revision

If the Hugging Face identifier resolves to a commit hash, capture that commit hash if possible.

Do not silently overwrite a user-specified revision.

Prefer pinned revisions in production/research configuration.

---

# 8. Task abstraction

Define something conceptually like:

```python
class Task(ABC):

    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def load(
        self,
        split: str,
        limit: int | None = None,
    ):
        ...

    def build_prompt(self, example) -> str:
        ...

    def parse_answer(self, output: str):
        ...

    def score(self, prediction, example) -> float:
        ...
```

However, preferably keep parsing/scoring as separate components if that makes the interfaces cleaner.

The important point is that the task owns benchmark semantics.

---

# 9. GSM8K implementation

Implement GSM8K using the Hugging Face `datasets` library.

Use the standard GSM8K test split.

Do not silently use a different GSM8K variant.

Store the dataset identifier and revision in the run metadata.

Each example should have a stable benchmark identifier.

If the source dataset does not provide a convenient stable identifier, generate one deterministically from the example index/content and document the strategy.

Support:

```text
split = test
limit = N
```

where `limit=None` means the complete split.

---

# 10. Prompting

The prompting system must be separated from the task.

Create a prompt-template abstraction.

For the initial GSM8K baseline, implement one explicitly documented protocol.

Use a consistent reasoning prompt such as:

```text
Solve the following math problem step by step.

{question}

Give the final answer clearly.
```

But choose one exact prompt and freeze it in the configuration.

Do not allow the baseline and future intervention models to accidentally use different prompt construction.

For instruct/chat models, correctly use the tokenizer's chat template when appropriate.

Prompt construction must be deterministic.

Save the actual rendered prompt for every example.

---

# 11. Generation configuration

Create a dedicated immutable configuration object, e.g.:

```python
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    seed: int
```

For the initial GSM8K baseline use deterministic generation:

```text
temperature = 0
do_sample = false
top_p = 1.0
```

Use a configurable but reasonable:

```text
max_new_tokens = 512
```

Do not introduce self-consistency yet.

Do not introduce multiple sampled trajectories yet.

---

# 12. Batch size

The evaluator should operate on batches.

Make batch size configurable:

```text
batch_size = 1
```

for initial MPS/CPU validation, with the ability to increase it later.

Do not hardcode batch size inside the model adapter.

The evaluator should determine batching and call the adapter with batches.

Ensure that changing batch size does not affect the logical ordering of predictions.

---

# 13. Answer parsing

Create a separate parser interface:

```python
class AnswerParser(ABC):

    def parse(self, text: str) -> ParsedAnswer:
        ...
```

Implement:

```text
GSM8KAnswerParser
```

The parser should robustly extract the final numerical answer from the generated reasoning.

The parsing policy must be:

* deterministic
* documented
* unit-tested
* versioned

Do not silently introduce arbitrary heuristics.

Capture whether parsing succeeded.

Every prediction record should include:

```text
parse_success
parsed_answer
```

---

# 14. GSM8K scoring

Create a scorer interface.

For GSM8K use exact-match correctness after normalization.

The result for each example should contain:

```text
correct = true/false
```

The aggregate metrics should include at minimum:

```text
accuracy
parse_rate
num_examples
num_correct
num_parse_failures
```

Accuracy should be:

```text
num_correct / num_examples
```

Do not silently exclude parse failures from the denominator.

---

# 15. Per-example result records

THIS IS REQUIRED.

Save every individual prediction.

Use JSONL.

Each record should include at least:

```json
{
  "example_id": "...",
  "prompt": "...",
  "raw_output": "...",
  "parsed_answer": "...",
  "reference_answer": "...",
  "parse_success": true,
  "correct": true
}
```

Also attach enough metadata to establish the relationship between this output and the exact experiment.

Do not merely save aggregate accuracy.

Future research analysis will depend on comparing individual baseline vs Recirculation outputs.

---

# 16. Paired comparison support

Design the output format so later we can perform:

```text
baseline correct → recirculation correct
baseline correct → recirculation wrong
baseline wrong   → recirculation correct
baseline wrong   → recirculation wrong
```

without rerunning the models.

Every example must therefore have a stable `example_id`.

The order of examples must be stable.

---

# 17. Run directory

Every evaluation invocation should create a unique run directory.

Use a layout similar to:

```text
results/
└── gsm8k/
    └── smollm2-360m-instruct/
        └── run_YYYYMMDD_HHMMSS_<shortid>/
            ├── manifest.json
            ├── config.yaml
            ├── metrics.json
            ├── predictions.jsonl
            ├── environment.json
            └── logs.txt
```

Do not overwrite previous runs.

---

# 18. Required reproducibility metadata

EVERY run must capture ALL of the following:

```text
Git commit
Python version
PyTorch version
Transformers version
CUDA version
GPU model
Model revision
Dataset revision
Tokenizer revision
dtype
seed
generation config
batch size
prompt template
task version
evaluation code version
```

This list is mandatory.

Also capture the following where available:

```text
Operating system
machine architecture
device
hostname or machine identifier if useful and non-sensitive
MPS availability
CUDA availability
GPU count
CPU information
RAM
timestamp
timezone
command used
```

Do not fail on environments where CUDA or GPU information is unavailable.

Represent unavailable fields explicitly as:

```json
null
```

or a documented value such as:

```text
"not_available"
```

Do not fabricate values.

---

# 19. Git metadata

At minimum record:

```text
git_commit
git_branch
git_dirty
```

Also record the command used to launch the evaluation.

If the repository has uncommitted changes, do NOT fail automatically.

Instead record:

```text
git_dirty = true
```

and optionally capture a short warning.

The goal is traceability, not artificial cleanliness.

---

# 20. Python / package metadata

Capture:

```text
python_version
torch_version
transformers_version
datasets_version
```

Also capture other major runtime dependencies that materially affect execution.

Do not hardcode these versions.

Read them from the active Python environment.

---

# 21. CUDA / device metadata

Capture:

```text
cuda_available
cuda_version
gpu_model
gpu_count
device_type
```

On Apple Silicon:

```text
cuda_available = false
device_type = mps
gpu_model = appropriate discovered Apple GPU information
```

Do not assume `nvidia-smi` exists.

Hardware discovery must be platform-aware.

---

# 22. Reproducibility utilities

Create a dedicated module that collects runtime metadata.

Conceptually:

```python
collect_environment_metadata() -> dict
```

It should be reusable independently of the evaluator.

Do not scatter environment probing across the codebase.

---

# 23. Prompt template versioning

Treat prompts as part of the experiment.

Define:

```text
prompt_template_name
prompt_template_version
prompt_template_hash
```

The rendered prompt should also be saved per example.

Changing the prompt should therefore change the experiment identity.

Use a deterministic hash of the canonical template/configuration.

---

# 24. Task versioning

GSM8K should have an explicit task version.

Example:

```text
task = gsm8k
task_version = "1.0"
```

Increment the task version when benchmark semantics change, such as:

* prompt changes
* answer parsing changes
* scoring changes
* dataset interpretation changes

Document this in:

```text
docs/evaluation_protocol.md
```

---

# 25. Evaluation code version

Define a code/evaluator version separate from the git commit.

For example:

```text
evaluation_code_version = "0.1.0"
```

The git commit provides exact source traceability.

The semantic evaluation version communicates whether evaluation semantics changed.

---

# 26. Config system

Use YAML or TOML configuration.

Do not scatter experiment parameters across CLI flags and source code.

A configuration should support:

```yaml
model:
  name: HuggingFaceTB/SmolLM2-360M-Instruct
  revision: null
  tokenizer_revision: null
  dtype: float32
  device: auto

task:
  name: gsm8k
  version: "1.0"
  split: test
  limit: 10

prompt:
  template_name: gsm8k_cot_v1
  template_version: "1.0"

generation:
  max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  seed: 42

runtime:
  batch_size: 1

experiment:
  name: gsm8k_baseline_smollm2_debug
```

Do not make the model implementation depend directly on this YAML structure.

Parse YAML into typed configuration objects.

---

# 27. CLI

Provide a CLI such as:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml
```

Also support simple overrides such as:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml \
  --limit 5 \
  --batch-size 1
```

Make the CLI print:

```text
Run ID
Model
Task
Split
Examples
Device
Generation configuration
Output directory
```

before execution.

---

# 28. Dry-run mode

Implement:

```bash
python scripts/evaluate.py \
  --config ... \
  --dry-run
```

Dry-run should:

* load the config
* resolve model/task
* inspect environment
* verify required dependencies
* display the resolved configuration
* validate the output directory
* validate prompt generation on one example
* NOT perform expensive generation

This is particularly important because the first environment is a Mac and the eventual environment is Modal.

---

# 29. Validation mode

Also create a lightweight validation mode.

For example:

```bash
python scripts/evaluate.py \
  --config ... \
  --limit 3
```

The validation run should exercise the COMPLETE pipeline:

```text
dataset
→ task
→ prompt
→ tokenizer
→ model
→ generation
→ parser
→ scorer
→ artifact writing
```

but only use a tiny number of examples.

---

# 30. Tests

This is a research infrastructure project.

Write tests before declaring it complete.

## Unit tests

Test:

* configuration parsing
* prompt rendering
* prompt hashing
* GSM8K answer parsing
* GSM8K scoring
* environment metadata collection
* manifest generation
* result serialization
* batching logic
* stable example IDs

## Integration tests

Test:

```text
dataset → evaluator → model adapter → parser → scorer → artifacts
```

with a tiny dataset.

Do not make the integration suite depend on downloading a huge model.

---

# 31. Mock model adapter

Create a deterministic fake model adapter for tests.

For example:

```python
MockModelAdapter
```

which returns predefined outputs based on example index/prompt.

This allows the evaluator to be tested without loading a real model.

This is important.

Most of the evaluation framework should be testable independently of GPU/model execution.

---

# 32. Tiny synthetic GSM8K-like fixture

Create a small local fixture containing 2–5 examples.

Use it for parser/scoring/integration tests.

Do not require internet access for unit tests.

A developer should be able to run:

```bash
pytest
```

without downloading the full GSM8K dataset or model.

---

# 33. Model download/integration test separation

Separate:

```text
offline tests
```

from:

```text
model-backed integration tests
```

The ordinary test suite must remain fast.

A model-backed smoke test may be explicitly invoked, e.g.:

```bash
pytest -m model
```

or via a separate script.

---

# 34. Dependency design

Use sensible dependencies, likely including:

```text
torch
transformers
datasets
pyyaml
pydantic
pytest
```

Do not add libraries just because they are convenient.

Keep the core lightweight.

Use standard Python libraries where practical.

---

# 35. Logging

Implement structured logging.

During execution log:

```text
run started
resolved model
resolved device
dataset loaded
N examples
generation batch i/N
results written
metrics computed
run completed
```

Avoid dumping every generated response to stdout.

Raw generations belong in `predictions.jsonl`.

---

# 36. Failure handling

The evaluator should fail loudly on genuinely invalid configurations.

For example:

```text
missing model
missing task
invalid generation configuration
invalid split
```

But individual example-level parsing failures should NOT crash the complete run.

Instead:

```text
parse_success = false
correct = false
```

and continue.

Record the failure.

---

# 37. Ordering guarantees

The framework must preserve:

```text
example order
```

through:

```text
dataset
→ prompts
→ batches
→ model outputs
→ result records
```

Changing batch size must not change which output belongs to which example.

Test this.

---

# 38. Determinism test

Create a smoke test where the same deterministic configuration is executed twice against the mock model.

The resulting logical predictions should be identical.

For the actual HF model, create documentation explaining which deterministic guarantees are expected and any platform-specific limitations, especially on MPS.

Do not falsely claim bit-for-bit reproducibility where the backend does not guarantee it.

---

# 39. Batch-size invariance

This is mandatory.

Run the same small fixture through:

```text
batch_size = 1
batch_size = 2
batch_size = 4
```

and verify:

```text
example_id → output
```

mapping remains correct.

This is a crucial correctness test before later running custom Recirculation logic.

---

# 40. Results schema

Define a versioned schema.

For example:

```text
result_schema_version = "1.0"
```

Manifest:

```json
{
  "result_schema_version": "1.0",
  "run_id": "...",

  "experiment": {
    "name": "...",
    "evaluation_code_version": "...",
    "command": "..."
  },

  "git": {
    "commit": "...",
    "branch": "...",
    "dirty": false
  },

  "model": {
    "name": "...",
    "revision": "...",
    "tokenizer_revision": "...",
    "dtype": "..."
  },

  "task": {
    "name": "gsm8k",
    "version": "1.0",
    "split": "test",
    "dataset_revision": "..."
  },

  "prompt": {
    "name": "...",
    "version": "...",
    "hash": "..."
  },

  "generation": {
    "seed": 42,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": false,
    "max_new_tokens": 512
  },

  "runtime": {
    "python_version": "...",
    "torch_version": "...",
    "transformers_version": "...",
    "cuda_version": null,
    "gpu_model": "...",
    "device": "mps",
    "batch_size": 1
  }
}
```

The precise field names may be improved, but all required information must be present.

---

# 41. No hidden global state

Do not use module-level global variables for:

* model
* tokenizer
* task
* generation settings
* random seeds
* output paths

Pass dependencies explicitly.

This will make future experimentation and unit testing much easier.

---

# 42. Avoid premature abstractions

Do NOT build:

* distributed evaluation
* multi-GPU scheduling
* Ray
* Kubernetes
* Modal-specific code
* vLLM integration
* safety benchmarks
* activation hooks
* Recirculation implementation
* logging into every intermediate transformer layer

yet.

We want a clean foundation.

Make the abstractions extensible, but implement only what is currently needed.

---

# 43. Future Recirculation compatibility

Even though Recirculation is NOT being implemented now, design the model interface so a later adapter can expose configuration such as:

```yaml
intervention:
  type: recirculation
  source_layer: 18
  destination_layer: 9
  alpha: 0.15
  normalization: destination_norm
```

But for now:

```yaml
intervention:
  type: none
```

The evaluator should treat intervention configuration as model metadata, not benchmark logic.

The future Recirculation adapter should be able to reuse:

```text
same Task
same prompts
same dataset
same scoring
same artifact format
same evaluator
```

with only the inference adapter changing.

---

# 44. Research comparison compatibility

Eventually we want to run:

```python
evaluator.evaluate(task=gsm8k, model=baseline)
evaluator.evaluate(task=gsm8k, model=recirculation)
evaluator.evaluate(task=gsm8k, model=other_intervention)
```

and obtain directly comparable artifacts.

Do not design the result system around "baseline" as a special case.

Treat baseline as:

```text
ModelAdapter with intervention_type = none
```

---

# 45. Independent evaluation cross-check

Do not implement the external benchmark cross-check yet unless it is cheap and clean.

However, document a future validation step using an established evaluator such as `lm-evaluation-harness`.

The purpose will be:

```text
our GSM8K result
vs
independent reference evaluator
```

under the same prompting/decoding protocol.

The local evaluator is not considered "trusted" simply because it runs.

We need to independently validate the benchmark implementation before using it for research claims.

---

# 46. Documentation

Create:

```text
README.md
docs/evaluation_protocol.md
```

README should explain:

* installation
* running tests
* running dry-run
* running a 5-example smoke test
* running a larger evaluation
* where results are written

`evaluation_protocol.md` should explicitly document:

```text
Dataset
Dataset revision
Split
Prompt template
Prompt version
Chat template behavior
Decoding settings
Max new tokens
Answer extraction
Scoring
Aggregation
Seed
Task version
Result schema
```

Treat this document as the benchmark contract.

---

# 47. Initial acceptance test

After implementation, execute the following workflow:

### Step 1

Run:

```bash
pytest
```

All offline tests should pass.

### Step 2

Run a dry run:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml \
  --dry-run
```

Verify that:

* configuration resolves
* model is under 500M parameters
* device is detected correctly
* all metadata fields can be collected
* prompt rendering works

### Step 3

Run a tiny real evaluation:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_smollm2_360m_debug.yaml \
  --limit 5
```

### Step 4

Verify the resulting run directory contains:

```text
manifest.json
config.yaml
metrics.json
predictions.jsonl
environment.json
logs.txt
```

### Step 5

Inspect `predictions.jsonl`.

Every example must contain:

```text
example_id
prompt
raw_output
parsed_answer
reference_answer
parse_success
correct
```

### Step 6

Run the same tiny evaluation with:

```text
batch_size=1
batch_size=2
```

and verify the example-to-output mapping is identical.

---

# 48. Expected CLI output

Aim for something similar to:

```text
============================================================
LLM Evaluation Harness
============================================================

Run ID:              20260905_XXXXXX_ab12
Task:                gsm8k
Task version:        1.0
Split:               test
Examples:            5

Model:               HuggingFaceTB/SmolLM2-360M-Instruct
Model revision:     <resolved revision>
Tokenizer revision: <resolved revision>

Device:              mps
DType:               float32

Generation:
  temperature:       0.0
  top_p:             1.0
  do_sample:         false
  max_new_tokens:    512
  seed:               42

Batch size:           1

Git commit:           <sha>
Git dirty:            false

Output:
results/gsm8k/smollm2-360m-instruct/run_...

------------------------------------------------------------
Evaluation
------------------------------------------------------------

[1/5] ...
[2/5] ...
...

------------------------------------------------------------
Results
------------------------------------------------------------

Examples:             5
Correct:              3
Accuracy:             0.6000
Parse rate:           1.0000

Run artifacts:
results/...
```

Do not hardcode the example numbers.

---

# 49. Code quality requirements

Use:

* type hints
* dataclasses or Pydantic models where appropriate
* clear docstrings on public interfaces
* small functions
* dependency injection where useful
* no unnecessary global state
* meaningful exceptions
* unit tests for parsing and scoring
* deterministic serialization where practical

Avoid:

* giant classes
* giant scripts
* benchmark-specific conditionals inside the evaluator
* model-specific conditionals inside the task
* magic constants
* hidden configuration
* silently changing experimental settings

---

# 50. Important future-facing requirement

Make sure a future custom model adapter can access generation internals.

The current interface is:

```python
generate(prompts, config) -> outputs
```

but design it so a future Recirculation adapter can implement custom generation while preserving the same contract.

Do not force all models into a `pipeline()` abstraction.

For example, an eventual adapter may need to call `model.forward()` token-by-token and maintain recurrent state.

The rest of the evaluator must remain unaware of this.

---

# 51. Do not optimize prematurely

The initial goal is:

```text
correctness > speed
```

We are validating infrastructure.

Readable and testable code is more important than squeezing maximum throughput from MPS.

Later, GPU deployment can optimize:

* batching
* attention implementation
* inference engine
* compilation
* KV-cache behavior
* GPU utilization

without changing the evaluation semantics.

---

# 52. Deliverables

At the end of the implementation provide:

1. Complete repository structure.
2. Working GSM8K task.
3. Working Hugging Face model adapter.
4. Model-independent evaluator.
5. Deterministic generation configuration.
6. Answer parser.
7. GSM8K scorer.
8. Per-example JSONL output.
9. Run manifest.
10. Environment metadata collection.
11. YAML configuration.
12. CLI.
13. Dry-run mode.
14. Tiny real GSM8K smoke test.
15. Mock model tests.
16. Batch-size invariance tests.
17. Documentation.
18. Commands needed to run everything on an M2 Mac.

Do not implement Recirculation yet.

Do not implement safety benchmarks yet.

Do not run a full GSM8K evaluation yet unless it is trivial after the smoke test.

---

# 53. Final acceptance criterion

Consider the task successful only if I can later replace:

```text
HFCausalLMAdapter
```

with:

```text
RecirculationModelAdapter
```

and run:

```bash
python scripts/evaluate.py \
  --config configs/gsm8k_<future_model>.yaml
```

while preserving exactly the same:

```text
dataset
prompting
benchmark semantics
answer parser
scoring
metrics
result schema
reproducibility metadata
per-example output format
```

The only intended experimental difference should be the model/inference intervention.

That property is the central design requirement of this project.
