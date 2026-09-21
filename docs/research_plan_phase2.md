# PHASE 2 — TRUSTED RESEARCH BASELINE + RECIRCULATION REPRODUCTION

You are extending the research-grade evaluation harness built in Phase 1.

Phase 1 established:

* modular task interfaces
* model adapters
* GSM8K task support
* prompting
* parsing
* scoring
* batching
* per-example prediction logging
* reproducibility manifests
* environment metadata
* unit/integration tests
* local execution on Apple Silicon

The purpose of Phase 2 is to transition from **infrastructure validation** to the first real research experiments.

The broader research project investigates whether **deep-to-shallow representation Recirculation** can improve model capability and eventually LLM safety.

The immediate goal of this phase is:

> Establish a trustworthy GSM8K baseline on the actual research model, implement the original fixed/training-free Recirculation mechanism as a new model adapter/intervention, reproduce the reported capability effect on GSM8K, and make the resulting comparison fully reproducible.

Do NOT implement the safety experiments yet.

Do NOT implement adaptive Recirculation yet.

Do NOT optimize for safety yet.

The output of this phase should be a clean, experimentally defensible foundation for Phase 3, where the same framework will be extended to safety evaluation.

---

# 1. Research context

The current Recirculation paper is:

```text
arXiv:2608.17981
Recirculation
current version: v2
submitted/revised: August 28, 2026
```

Use the **latest v2 as the canonical implementation/reference document**, not the original v1.

The paper describes Recirculation as an inference-time architectural intervention in which an activation from a deeper source layer at one input step is mixed into a shallower destination layer for the next input step.

Conceptually:

```text
current token t
      │
      ▼
normal transformer computation
      │
      ├──────────────► shallow destination state
      │
      ▼
deep source activation
      │
      ▼
recirculation
      │
      ▼
shallow destination at next step t+1
```

The paper defines the update approximately as:

```text
d' = α f(s) + β d
```

where:

* `s` = deep source residual representation from the current step
* `d` = destination residual representation
* `d'` = modified destination state used in the next recurrent step
* `α` = source mixing coefficient
* `β` = destination coefficient
* `f(.)` = source normalization/transformation

For the principal normalization, the source is rescaled to the destination L2 norm:

```text
f(s) = ||d||₂ / ||s||₂ * s
```

The paper reports that the useful behavior depends on source/destination depth and the mixture coefficients, and explicitly distinguishes Recirculation from simply looping transformer layers.

The paper reports the following source/destination pairs from its perplexity tuning:

```text
Gemma3 1B PT   source=11 destination=4
Gemma3 4B PT   source=18 destination=9
Gemma3 12B PT  source=35 destination=16
```

The paper uses these selected pairs for subsequent evaluation and reports that the 4B GSM8K evaluation shows improvement for both greedy pass@1 and sampled pass@128.

---

# 2. Phase 2 scientific objective

There are THREE separate objectives in this phase.

## Objective A — Trusted GSM8K baseline

Establish the baseline performance of the actual research model under a frozen evaluation protocol.

The result must be independently reproducible.

## Objective B — Correct Recirculation implementation

Implement the fixed training-free Recirculation mechanism without modifying model weights.

The implementation must preserve the existing evaluator contract.

## Objective C — Reproduction

Compare:

```text
Baseline
vs
Baseline + fixed Recirculation
```

on GSM8K.

The first goal is not to beat the paper.

The first goal is:

> determine whether our implementation produces the same qualitative phenomenon as the paper under a clearly documented protocol.

Do not tune the method to force agreement.

---

# 3. Model progression

Use the following progression.

## Stage 2A — Gemma3 1B PT

Use:

```text
Gemma3 1B PT
```

first.

Purpose:

* validate Recirculation implementation
* inspect activations
* debug token-to-token state transfer
* verify layer indexing
* verify normalization
* verify batch behavior
* measure runtime
* run small GSM8K smoke tests

Do NOT immediately run the entire GSM8K benchmark.

Start with a small deterministic subset.

---

## Stage 2B — Gemma3 4B PT

Once 1B is validated, use:

```text
Gemma3 4B PT
```

as the primary research model for the phase.

This is the most important model because the Recirculation paper specifically reports GSM8K results for Gemma3 4B PT.

Use the paper's reported 4B source/destination pair:

```text
source_layer = 18
destination_layer = 9
```

as the initial fixed Recirculation configuration.

---

## Stage 2C — Gemma3 12B PT

Do NOT require this in the initial Phase 2 implementation.

The adapter and configuration system must support it, but execution should be optional.

Only run 12B after 1B and 4B are working correctly.

---

# 4. Critical distinction: PT vs IT

The Recirculation paper distinguishes pretrained models from instruction-tuned models and reports its standard GSM8K experiment using Gemma3 PT models.

Therefore Phase 2 should primarily use:

```text
Gemma3 PT
```

not IT.

Do not silently substitute an instruction-tuned checkpoint.

The model configuration must explicitly encode:

```yaml
model:
  family: gemma3
  size: 4b
  tuning: pretrained
```

Later phases may introduce IT models for safety experiments.

---

# 5. Do NOT change the Phase 1 evaluator architecture

The existing architecture must remain:

```text
Task
   ↓
Prompt
   ↓
ModelAdapter
   ↓
Generation
   ↓
Parser
   ↓
Scorer
   ↓
Results
```

Do NOT put Recirculation logic inside:

* GSM8K task
* evaluator
* scorer
* parser
* benchmark code

Implement Recirculation in the model/inference layer.

Conceptually:

```text
HFCausalLMAdapter
       │
       ├── baseline generation
       │
       └── recirculation generation
```

or preferably:

```text
HFCausalLMAdapter
       ↑
ModelAdapter
       ↑
Intervention-capable generation
       ↑
RecirculationIntervention
```

Use the cleanest design consistent with the existing Phase 1 abstractions.

---

# 6. Introduce an intervention abstraction

The harness should now support a first-class concept of an inference intervention.

Create something conceptually equivalent to:

```python
class Intervention(ABC):

    @property
    def name(self) -> str:
        ...

    def modify_generation(
        self,
        model,
        ...
    ):
        ...
```

However, do not force all interventions to have an identical low-level implementation.

The important architectural requirement is:

```text
Benchmark
    ↓
Evaluator
    ↓
Model Adapter
    ↓
Intervention
```

The benchmark remains completely unaware of the intervention.

---

# 7. Baseline must remain a first-class experimental condition

Represent baseline explicitly as:

```yaml
intervention:
  type: none
```

Do NOT special-case baseline as a completely different evaluation pathway.

The eventual research matrix should be able to look like:

```text
baseline
recirculation
looping
random-recirculation
other-intervention
```

using exactly the same evaluator.

---

# 8. Recirculation configuration

Create an explicit typed configuration.

For example:

```yaml
intervention:
  type: recirculation

  source_layer: 18
  destination_layer: 9

  alpha: 0.15
  beta: 1.0

  normalization:
    type: destination_l2

  ramping:
    enabled: false
```

Do not scatter these parameters through implementation code.

The manifest must record every Recirculation-specific parameter.

---

# 9. Start with the fixed training-free variant

Do NOT implement adaptive Recirculation yet.

Do NOT train an MLP.

Do NOT fine-tune the model.

Do NOT learn alpha/beta.

Start with the simple fixed mixture.

The objective is to establish:

```text
baseline
vs
fixed recirculation
```

before adding learned components.

The paper itself distinguishes the basic fixed method from its later adaptive variant, and describes adaptive Recirculation as a separate method involving learned conditional vector-valued coefficients.

---

# 10. Implement normalization explicitly

Implement destination-norm source scaling:

```python
scaled_source = (
    source
    * destination.norm(...)
    / source.norm(...)
)
```

Use numerically safe division.

Do not assume source and destination norms are nonzero.

Use:

```text
eps
```

where appropriate.

Log:

```text
source_norm
destination_norm
scaled_source_norm
```

for debug runs.

Do not log full activation tensors by default.

---

# 11. Implement both beta conventions

The paper reports that the preferred mixture differs by model size.

For Gemma3 1B, a convex mixture is used:

```text
beta = 1 - alpha
```

while for Gemma3 4B and 12B the paper reports that the non-convex:

```text
beta = 1
```

setting is important.

Therefore the implementation must support:

```yaml
mixture:
  mode: convex
```

and

```yaml
mixture:
  mode: nonconvex
```

or explicit `beta`.

Do not hardcode:

```python
beta = 1 - alpha
```

inside the algorithm.

---

# 12. Initial 1B configuration

For Gemma3 1B PT, create a reproduction configuration based on the paper:

```yaml
model:
  name: <Gemma3 1B PT>
  
intervention:
  type: recirculation

  source_layer: 11
  destination_layer: 4

  alpha: 0.15
  beta: 0.85

  normalization:
    type: destination_l2
```

The paper reports `11 -> 4` as the selected 1B source/destination pair from its tuning procedure.

Do not claim exact reproduction of their final numbers unless the prompting, dataset preparation, model revision, and evaluation protocol are demonstrably matched.

---

# 13. Initial 4B configuration

For Gemma3 4B PT:

```yaml
model:
  name: <Gemma3 4B PT>

intervention:
  type: recirculation

  source_layer: 18
  destination_layer: 9

  alpha: 0.15
  beta: 1.0

  normalization:
    type: destination_l2
```

The pair `18 -> 9` is reported in the current paper as the source/destination pair selected through the perplexity tuning procedure.

---

# 14. VERY IMPORTANT: clarify what is being reproduced

There are three different experiments that must not be conflated.

## Experiment R1

Paper configuration + our implementation.

Question:

> Does the fixed Recirculation mechanism work in our implementation?

## Experiment R2

Paper configuration + our GSM8K evaluation.

Question:

> Does the capability effect reproduce qualitatively?

## Experiment R3

Our own layer/alpha sweep.

Question:

> Does the model exhibit a similar optimization landscape independently?

Do not merge these.

---

# 15. GSM8K protocol

Freeze one evaluation protocol before running comparisons.

Use:

```text
dataset = GSM8K
split = test
prompting = zero-shot CoT
```

The Recirculation paper describes its GSM8K evaluation as zero-shot and uses a zero-shot chain-of-thought prompt. It evaluates greedy pass@1 and sampled pass@128 separately.

For Phase 2, implement **pass@1 first**.

Do not implement pass@128 until deterministic pass@1 is working.

---

# 16. Phase 2 pass@1

Initial generation configuration:

```yaml
generation:
  do_sample: false
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 512
```

The primary metric:

```text
accuracy
```

Secondary metrics:

```text
parse_rate
num_correct
num_incorrect
num_parse_failures
```

Use exactly the same configuration for baseline and Recirculation.

---

# 17. Do not tune on GSM8K test

For the first reproduction experiment:

```text
NO layer sweep on GSM8K test
NO alpha sweep on GSM8K test
NO prompt tuning on GSM8K test
```

Use the paper's published parameters first.

The purpose is to obtain a clean reproduction rather than an optimized GSM8K score.

Later, a separate development split can be introduced for our own hyperparameter search.

---

# 18. Run order

Execute the following sequence.

## Run 1

Gemma3 1B PT baseline.

Use a tiny subset:

```text
5 examples
```

Purpose:

* verify model loading
* verify tokenizer
* verify prompt
* verify output
* verify parser
* verify manifest

---

## Run 2

Gemma3 1B PT Recirculation.

Same 5 examples.

Compare:

```text
example_id
baseline output
recirculated output
```

Do not care about statistical significance at this stage.

This is an implementation smoke test.

---

## Run 3

Increase to:

```text
50 examples
```

Run:

```text
baseline
recirculation
```

Check:

* no crashes
* ordering preserved
* batch-size invariance
* output artifacts correct
* Recirculation is actually being invoked

---

## Run 4

Run the complete GSM8K test set on the 1B model.

Produce:

```text
baseline accuracy
recirculation accuracy
delta
```

Save every prediction.

---

## Run 5

Gemma3 4B PT baseline.

Start with:

```text
10 examples
```

Then:

```text
100 examples
```

Then full GSM8K.

---

## Run 6

Gemma3 4B PT + fixed Recirculation.

Use:

```text
source = 18
destination = 9
alpha = 0.15
beta = 1.0
destination-L2 normalization
```

Then compare against the 4B baseline.

---

# 19. Paired evaluation

The comparison must be performed at the example level.

The result analysis should automatically generate:

```text
baseline correct
→ recirculation correct

baseline correct
→ recirculation wrong

baseline wrong
→ recirculation correct

baseline wrong
→ recirculation wrong
```

For GSM8K test set:

```python
transition_matrix(...)
```

should return a table like:

```text
                    Recirc
                 Wrong  Correct
Baseline Wrong      A       B
Baseline Correct    C       D
```

Then report:

```text
baseline_accuracy
recirc_accuracy
absolute_delta
relative_delta
fixed_correct
fixed_incorrect
rescued_examples
regressed_examples
```

This is a required research artifact.

---

# 20. Example-level diff

Create an analysis utility:

```bash
python scripts/compare_runs.py \
    --baseline <run> \
    --treatment <run>
```

It should produce:

```text
comparison.json
comparison.csv
summary.json
```

and optionally a human-readable report.

For every changed example show:

```text
example_id
question
reference_answer
baseline_parsed_answer
recirc_parsed_answer
baseline_correct
recirc_correct
baseline_output
recirc_output
```

This will later be extremely useful for understanding why Recirculation changes reasoning behavior.

---

# 21. Add answer-transition analysis

Compute:

$$
\Delta accuracy =
\frac{N_{wrong\rightarrow correct}
-
N_{correct\rightarrow wrong}}
{N}
$$

as a descriptive analysis.

Do not replace ordinary accuracy with this.

Both must be reported.

---

# 22. Generation-length analysis

For GSM8K, compare baseline and Recirculation on:

```text
output token count
reasoning trace length
answer position
parse failures
```

Create:

```text
mean output length
median output length
p90 output length
```

and compare conditions.

This is not a primary metric.

It is diagnostic.

The research question is whether Recirculation changes reasoning trajectories, not merely answer strings.

---

# 23. First Recirculation correctness invariant

Implement a debug mode where the model generates using:

```text
alpha = 0
```

This should behave as:

```text
baseline
```

up to expected numerical/backend effects.

This is a crucial test.

Run:

```text
baseline
vs
recirculation(alpha=0)
```

on the same examples.

Expected result:

```text
identical predictions
```

under deterministic evaluation, subject to documented backend numerical behavior.

If this fails, investigate before testing nonzero alpha.

---

# 24. Second invariant: source/destination identity check

Create an artificial configuration where:

```text
source_layer == destination_layer
```

should either be rejected as invalid or handled explicitly according to the implementation contract.

Do not allow ambiguous layer configurations.

Validate:

```text
source_layer > destination_layer
```

for standard Recirculation.

---

# 25. Third invariant: no-intervention equivalence

The following configurations should produce equivalent logical behavior:

```text
intervention = none
```

and

```text
intervention = recirculation
alpha = 0
```

This should become an automated integration test.

---

# 26. Verify recurrence timing carefully

This is the most important implementation detail.

Do NOT accidentally implement:

```text
deep layer at t
→ shallow layer at t
```

inside the same forward pass.

That is not the intended Recirculation mechanism.

The intended mechanism is cross-step:

```text
step t:
    compute source activation at deep layer

step t+1:
    modify destination state using source activation from t
```

The recurrence therefore spans both:

```text
layer depth
```

and

```text
input step
```

The paper explicitly distinguishes this from looping, where recurrence occurs through depth rather than across successive input steps.

Document this carefully in the code.

---

# 27. Do not accidentally implement ordinary residual addition

Avoid implementing:

```python
destination += alpha * source
```

without accounting for:

* source/destination normalization
* beta
* step offset
* state storage
* correct layer boundary
* correct residual stream location

The implementation must map to the paper's intended recurrence semantics.

---

# 28. Add activation debug tracing

Create an optional debug mode:

```yaml
debug:
  capture_activations: true
  examples: 1
  token_positions: [0, 1, 2, 3]
```

For a tiny number of examples, capture:

```text
source norm
destination norm
scaled source norm
mixed destination norm
cosine(source,destination)
```

Do NOT capture all hidden states for full GSM8K runs.

The activation-debug system must be opt-in.

---

# 29. Validate layer numbering

Different frameworks may number:

* embeddings
* transformer blocks
* hidden states

differently.

Create an explicit layer-index mapping.

Document whether:

```text
layer 0
```

means:

```text
first transformer block output
```

or:

```text
embedding output
```

Do not assume this.

Cross-check the implementation against the paper's indexing convention.

This is critical for reproducing:

```text
11 → 4
18 → 9
35 → 16
```

---

# 30. KV-cache behavior

Document how the custom autoregressive generation interacts with the KV cache.

Recirculation introduces recurrent hidden state across steps.

Ensure the implementation does not accidentally:

* reuse stale hidden states
* reuse incompatible KV cache state
* modify past tokens unintentionally
* recompute the wrong context
* apply Recirculation to only generated tokens when the intended protocol requires it during prefill

Do not optimize this prematurely.

Correctness first.

---

# 31. Prefill vs decode

Measure the two phases separately:

```text
prefill latency
decode latency
total latency
```

The Recirculation paper states that it has essentially no additional generation-time latency under parallel execution but introduces serial processing of the prefill context.

Do not claim equivalent latency behavior until it has actually been measured in this implementation.

---

# 32. Performance instrumentation

For every model/inference run capture:

```text
wall_clock_seconds
prefill_seconds
generation_seconds
tokens_generated
input_tokens
output_tokens
tokens_per_second
peak_memory
```

Where hardware APIs allow it, capture peak device memory.

On MPS, do not pretend CUDA memory statistics exist.

Represent unavailable metrics explicitly.

---

# 33. Batch-size experiment

Run:

```text
batch_size=1
batch_size=2
batch_size=4
```

where the hardware allows.

For deterministic generation verify:

```text
example_id
→ output
```

mapping remains identical.

The benchmark score must not change simply because batch size changed.

This should be an explicit Phase 2 validation test.

---

# 34. Re-run reproducibility experiment

Run the same deterministic configuration twice.

For example:

```text
Gemma3 1B PT
GSM8K test
limit=50
seed=42
temperature=0
```

Run:

```text
run_A
run_B
```

Then automatically compare:

```text
raw outputs
parsed answers
correctness
metrics
```

and report differences.

The expected logical result should be identical.

If not identical, determine whether the difference comes from:

* model/backend nondeterminism
* MPS
* generation implementation
* random seed handling
* data ordering

Do not simply say "seed fixed."

---

# 35. Benchmark configuration must be immutable after release

Create a research protocol file:

```text
docs/phase2_gsm8k_protocol.md
```

It should freeze:

```text
model
model revision
tokenizer revision
dataset
dataset revision
split
prompt template
prompt version
chat template
generation settings
answer parser
scorer
source layer
destination layer
alpha
beta
normalization
seed
```

Once the official baseline has been run:

```text
DO NOT MODIFY
```

Instead create a new protocol version if anything changes.

---

# 36. Independent benchmark validation

Before making a research claim, compare the baseline against an independent evaluation implementation such as `lm-evaluation-harness`, using an equivalent GSM8K protocol where possible.

The Recirculation paper itself used evaluation-harness for several standard downstream tasks, although GSM8K is separately described in its downstream section.

The purpose is:

```text
validate evaluator
```

not:

```text
make our result match an external number at all costs
```

Document any protocol differences explicitly.

---

# 37. Do not blindly chase the paper's exact score

The paper's reported numbers can depend on:

* model revision
* prompting
* tokenization
* answer parsing
* decoding
* implementation details
* hardware/backend
* evaluation code
* version of the paper

Therefore the outcome should be classified as:

```text
Exact reproduction
```

only if the protocol is actually matched.

Otherwise use:

```text
Independent replication
```

and compare qualitative trends.

Do not fabricate an exact reproduction claim.

---

# 38. GSM8K pass@128 should be Phase 2B

Once pass@1 is fully validated, add:

```text
pass@128
```

This is useful because the paper explicitly distinguishes:

```text
pass@1
```

as capability sharpening and

```text
pass@128
```

as capability expansion.

Implement this as a separate generation/evaluation mode.

Do NOT mix pass@1 and pass@128 in one metric.

---

# 39. Pass@128 requirements

Implement:

```yaml
generation:
  do_sample: true
  temperature: <paper-compatible value>
  top_p: <paper-compatible value>

evaluation:
  num_samples: 128
```

Store:

```text
sample_id
example_id
raw_output
parsed_answer
correct
```

for every sample.

Then compute:

$$
pass@128
=
\frac{\#\text{examples with at least one correct sample}}
{\#\text{examples}}.
$$

Do not treat the 128 samples as 128 independent benchmark examples.

They belong to the same problem.

---

# 40. Keep pass@1 and pass@128 artifacts separate

For example:

```text
results/
└── gsm8k/
    └── gemma3-4b-pt/
        └── pass_at_1/
        └── pass_at_128/
```

Do not overwrite deterministic outputs with sampled outputs.

---

# 41. First layer sweep — do NOT optimize GSM8K yet

After fixed Recirculation works, create infrastructure for a layer sweep.

Do NOT immediately sweep the full GSM8K test set.

First use:

```text
small development subset
```

and verify the sweep machinery.

The sweep should vary:

```text
source_layer
destination_layer
alpha
```

while preserving:

```text
same dataset subset
same prompts
same seed
same generation settings
```

The paper performs structured source/destination sweeps and observes systematic regions rather than isolated random successes.

---

# 42. Sweep output format

Each configuration should produce a record:

```json
{
  "source_layer": 18,
  "destination_layer": 9,
  "alpha": 0.15,
  "beta": 1.0,
  "normalization": "destination_l2",
  "accuracy": 0.XX,
  "num_examples": 100
}
```

Aggregate into a dataframe/table for heatmap visualization.

---

# 43. No test leakage during sweep

Use:

```text
GSM8K development subset
```

for exploratory layer/alpha selection.

Keep the official GSM8K test result untouched.

Do not repeatedly inspect test performance and then modify the intervention based on it.

Once a configuration is selected:

```text
freeze configuration
```

then run the test set exactly once for the official result.

You may perform additional confirmatory runs later, but they should be documented as such.

---

# 44. Alpha sweep

Once layer-pair mechanics are validated, test a small development sweep:

```text
alpha =
0
0.03
0.05
0.07
0.10
0.15
0.20
```

Do not start with a huge combinatorial sweep.

The purpose is to understand the local response surface.

Record:

```text
accuracy
parse rate
output length
latency
```

---

# 45. Layer-pair sweep

For Gemma3 1B initially:

```text
source > destination
```

and reasonable layer spacing.

Do not brute-force every possible pair on GSM8K immediately.

Start with:

```text
source around middle/late layers
destination around shallow/middle layers
```

because the paper reports useful regions where deeper source representations are routed to shallower destinations.

The actual experiment should remain broad enough to discover failures as well as successes.

---

# 46. Heatmap outputs

Generate:

```text
accuracy(source, destination)
```

and:

```text
delta_accuracy(source, destination)
```

For fixed alpha.

Later generate:

```text
accuracy(source, destination, alpha)
```

as separate slices.

Do not only visualize the best configuration.

The complete landscape is scientifically important.

---

# 47. Baseline comparison must appear on every sweep

Every sweep report must reference the same baseline:

```text
baseline_accuracy
```

and calculate:

```text
delta = recirculation_accuracy - baseline_accuracy
```

Never compare each configuration to another Recirculation configuration.

---

# 48. Add regression detection

The analysis should identify:

```text
improved examples
regressed examples
unchanged examples
```

For every sweep configuration.

This will later help us determine whether a safety improvement comes from genuine discrimination or simply from changing behavior globally.

---

# 49. Statistical reporting

For the full GSM8K benchmark report:

```text
N
accuracy
95% confidence interval
```

For paired baseline vs Recirculation analysis, use an appropriate paired method for binary outcomes, such as a paired bootstrap or McNemar-style analysis.

Do not use an inappropriate independent-sample test simply because it is convenient.

For the initial small smoke tests, statistical significance is unnecessary.

They are implementation tests.

---

# 50. Store all experiment provenance

The Phase 1 manifest requirements remain mandatory.

Add these Recirculation-specific fields:

```text
intervention_type
source_layer
destination_layer
alpha
beta
normalization
ramping
recurrence_variant
layer_indexing_convention
```

Also capture:

```text
model_parameters
hidden_size
num_layers
context_length
```

where obtainable.

---

# 51. Research run naming

Use names such as:

```text
gsm8k_gemma3_1b_pt_baseline
gsm8k_gemma3_1b_pt_recirc_11_4_a015
gsm8k_gemma3_4b_pt_baseline
gsm8k_gemma3_4b_pt_recirc_18_9_a015_b1
```

Do not encode important metadata only in directory names.

The manifest remains the canonical source.

---

# 52. Required Phase 2 artifacts

At completion, the repository should contain:

```text
results/
  gsm8k/
    gemma3-1b-pt/
      baseline/
      recirculation/

    gemma3-4b-pt/
      baseline/
      recirculation/
```

Each run should contain:

```text
manifest.json
config.yaml
metrics.json
predictions.jsonl
environment.json
logs.txt
```

and comparison runs should additionally contain:

```text
comparison.json
comparison.csv
```

---

# 53. Required visualizations

Create an analysis script that can generate:

### Figure A

Baseline vs Recirculation GSM8K accuracy.

### Figure B

Baseline → Recirculation transition matrix.

### Figure C

Output length distribution.

### Figure D

Alpha sweep.

### Figure E

Source/destination layer heatmap.

Do not manually edit numbers into figures.

Figures must be generated directly from stored experiment artifacts.

---

# 54. Required analysis notebook/script

Create:

```text
analysis/gsm8k_phase2.py
```

or an equivalent clean analysis module.

It should load:

```text
manifest
metrics
predictions
```

and generate summary tables/plots.

Do not hardcode result numbers.

---

# 55. Phase 2 acceptance criteria

Phase 2 is complete only when ALL of the following are true.

## Infrastructure

The Phase 1 test suite still passes.

## Model

Gemma3 1B PT loads correctly.

Gemma3 4B PT loads correctly.

## Baseline

A complete GSM8K baseline can be executed through the Phase 1 evaluator.

## Recirculation

Fixed training-free Recirculation executes through a model adapter/intervention without changing evaluator code.

## Correctness

`alpha=0` Recirculation matches baseline behavior within documented backend tolerance.

## Recurrence

The implementation genuinely transfers:

```text
deep layer @ t
→ shallow layer @ t+1
```

rather than performing same-step layer mixing.

## Reproducibility

Repeated deterministic runs produce identical logical predictions.

## Batch invariance

Changing batch size does not change example-to-output mapping or the resulting deterministic metric.

## Logging

All required metadata are captured.

## Comparison

A baseline/Recirculation paired comparison can be produced without rerunning the models.

## Artifacts

Every run is immutable and independently inspectable.

## Research

At least one full GSM8K comparison exists for the 1B model.

At least one full GSM8K comparison exists for the 4B model, assuming the hardware environment is sufficient.

---

# 56. Final Phase 2 report

At the end of execution, generate:

```text
reports/phase2_gsm8k_report.md
```

The report must contain:

## 1. Objective

What was tested.

## 2. Exact protocol

Model, revision, prompt, generation, task version, etc.

## 3. Baseline

Baseline GSM8K result.

## 4. Recirculation

Configuration and result.

## 5. Difference

Absolute and relative change.

## 6. Paired transitions

```text
wrong → correct
correct → wrong
```

## 7. Robustness

Repeated-run and batch-size tests.

## 8. Runtime

Latency and memory.

## 9. Layer/alpha exploration

Only if the sweep was executed.

## 10. Reproduction assessment

Classify the outcome as:

```text
exact reproduction
partial reproduction
qualitative replication
implementation failure
```

based on evidence.

Do not force a positive conclusion.

## 11. Known discrepancies

Explicitly document differences from the paper, such as:

* model revision
* prompt format
* tokenizer
* decoding
* dataset revision
* implementation framework
* hardware
* layer indexing
* normalization convention

## 12. Recommendation for Phase 3

The report should state which experimental configurations are sufficiently validated to be carried into the safety experiments.

Do NOT automatically select the configuration solely because it obtained the highest GSM8K score.

---

# 57. Important research principle

Do not optimize the implementation toward the expected paper result.

The objective is:

```text
correct implementation
        ↓
reproducible baseline
        ↓
reproducible intervention
        ↓
honest measurement
```

not:

```text
paper result
     ↓
modify implementation
     ↓
force matching score
```

If your numbers differ substantially, investigate and document the discrepancy.

That itself is useful experimental information.

---

# 58. What Phase 2 must NOT contain

Do NOT implement:

```text
HarmBench
AdvBench
XSTest
OR-Bench
JBB
Safety judges
Refusal classifiers
Activation probes
Safety directions
Activation ablations
ALIGNBEAM
Combined Recirculation + ALIGNBEAM
Adaptive Recirculation
Fine-tuning
RL
```

Those belong to later phases.

Do not let safety-specific code leak into this phase.

---

# 59. Future compatibility

The final architecture should make Phase 3 trivial conceptually:

```text
Phase 2

GSM8K
   ↓
Baseline / Recirculation


Phase 3

Safety benchmarks
   ↓
Baseline / Recirculation
```

The evaluator, result schema, model adapter, experiment manifest, and logging system should be reused.

Only new tasks, metrics, and analysis modules should be introduced.

---

# 60. Final command sequence the agent must support

At minimum:

```bash
# Tests
pytest

# Dry run
python scripts/evaluate.py \
    --config configs/gsm8k_gemma3_1b_baseline.yaml \
    --dry-run

# Tiny baseline smoke test
python scripts/evaluate.py \
    --config configs/gsm8k_gemma3_1b_baseline.yaml \
    --limit 5

# Tiny Recirculation smoke test
python scripts/evaluate.py \
    --config configs/gsm8k_gemma3_1b_recirc.yaml \
    --limit 5

# Full baseline
python scripts/evaluate.py \
    --config configs/gsm8k_gemma3_1b_baseline.yaml

# Full Recirculation
python scripts/evaluate.py \
    --config configs/gsm8k_gemma3_1b_recirc.yaml

# Compare
python scripts/compare_runs.py \
    --baseline <baseline_run> \
    --treatment <recirc_run>
```

Adapt command names to the Phase 1 CLI if necessary rather than creating a competing CLI.

---

# 61. Required final response from the coding agent

When implementation is complete, report:

```text
1. Files changed
2. Tests added
3. Tests passed
4. Exact model used
5. Exact model revision
6. Exact GSM8K protocol
7. Exact Recirculation configuration
8. Smoke-test results
9. Full-evaluation results if executed
10. Output artifact locations
11. Any discrepancies from the paper
12. Any unresolved implementation concerns
```

Do not merely say:

```text
"Implementation complete."
```

Show the evidence.

---

# 62. Definition of done

The ultimate definition of done for Phase 2 is:

> We can take the same GSM8K prompts, send them through a frozen baseline model and a frozen-weight Recirculation model, obtain per-example predictions through the same evaluator, recover the same scores when repeated deterministically, inspect exactly which examples changed, and trace every reported number back to a versioned experiment manifest.

Only after this is true should Phase 3 begin.

Phase 3 will use this exact infrastructure to answer the actual research question:

```text
Does deep-to-shallow representation Recirculation
improve safety behavior while preserving utility?
```
