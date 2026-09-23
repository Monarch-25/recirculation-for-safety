# PHASE 3 — SAFETY EVALUATION AND CORE RESEARCH EXPERIMENT

You are extending the research-grade LLM evaluation harness from Phase 1 and the validated Recirculation/GSM8K pipeline from Phase 2.

Phase 1 established:

* modular evaluation architecture
* task abstraction
* model adapters
* prompting
* generation
* parsing
* scoring
* batching
* per-example artifacts
* reproducibility metadata
* experiment manifests
* deterministic execution
* unit/integration testing

Phase 2 established:

* GSM8K baseline
* research-model loading
* Gemma3 PT evaluation
* fixed training-free Recirculation
* cross-step deep-to-shallow recurrence
* source/destination layer configuration
* alpha/beta configuration
* normalization
* layer/alpha experimentation infrastructure
* paired baseline vs Recirculation comparison
* runtime instrumentation
* reproducibility and batch-invariance checks

The purpose of Phase 3 is to begin the **actual safety research experiment**.

The central research question is:

> **Does deep-to-shallow representation Recirculation improve safety behavior of an aligned LLM while preserving benign utility and reasoning capability?**

The primary intervention remains:

```text
frozen model weights
+
inference-time deep-to-shallow Recirculation
```

Do NOT fine-tune the model.

Do NOT train a safety adapter.

Do NOT learn an intervention.

Do NOT implement adaptive Recirculation yet.

Do NOT combine Recirculation with ALIGNBEAM yet.

This phase is about establishing the first credible behavioral safety result.

---

# 0. STATUS — capability-gate outcome binds this protocol (2026-09-23)

Read this section first; it updates the spec below.

- **Capability gate passed (within noise).** Paper-exact (`mozer`) replication
  on GSM8K-Platinum, n=1209: ours 531/1209 vs repo dense 540 (McNemar p=0.58)
  and repo recirc 554 (p=0.12). The repo's own +14 is not significant (p=0.20).
  Evidence: `reports/paper_gsm8k_mt_replication.md`,
  `docs/findings_mozer_comparison.md`.
- **`mozer` is the intervention, not `cross_step`.** Mozer et al.'s method is
  same-token replay + **first-pass readout**. Our prior `cross_step` schedule is
  the reference repo's *withdrawn* delayed-cross-token class — it may appear in
  Phase 3 only as an explicit, labeled control, never as "the Recirculation
  condition". Every "Fixed Recirculation" below means schedule `mozer`.
- **Statistical discipline is calibrated, not aspirational (§42, §43, §67).**
  Full-scale paired GSM8K deltas sit within ±2.5 pts at 80% power; a borderline
  p≈0.05 (+28 rows, ours) and p=0.20 (repo +14) both fail a
  multiple-comparison view. Phase 3 must report per-benchmark paired McNemar
  with exact CIs and hold the pre-registered comparison list constant before
  inspecting results.
- **Pipeline/controls already exist.** `schedule: mozer` is implemented and
  on-device validated (α=0 ≡ baseline bitwise); alpha-zero, random-pair and
  shuffle controls are wired; manifests pin model/dataset revision, template
  hash, parser version, stop strings and seed. The replication audit
  (multiturn prompting, model EOS set, per-row positions, extraction priority)
  is documented in `reports/paper_gsm8k_mt_replication.md §2.2`, so the safety
  phase inherits a gap-audited machine rather than a black box.
- **Published artifact.** Replication vitals (531/559, Wilson CIs, +28
  p=0.054, quirks list) are live on the site (gh-pages); the site is the
  public-facing record of the capability gate.

No section below is deleted. §21's paper-derived 4B pair (18→9, α=0.15,
β=1, Appendix B.3) remains the initial fixed configuration, now
unambiguously under the `mozer` schedule on the IT model.

---

# 1. Research hypothesis

The working hypothesis is:

```text
Safety-relevant contextual information may become more explicit
at deeper transformer layers.

Deep representation
        ↓
Recirculation
        ↓
shallow representation at a later step
        ↓
improved downstream safety behavior
```

Formally, if:

```text
h_s(t)
```

is the residual representation at deeper layer `s` and step `t`, and:

```text
h_d(t+1)
```

is the destination representation at a shallower layer `d` during the next step, Recirculation modifies:

```text
h_d(t+1)
```

using information from:

```text
h_s(t)
```

with fixed weights.

The behavioral hypothesis is:

```text
Δ harmful refusal > 0
```

while:

```text
Δ benign over-refusal ≈ 0
```

and ideally:

```text
Δ benign utility ≈ 0
```

or better.

Do NOT assume these hypotheses are true.

The experiment must be able to produce:

```text
positive safety effect
neutral effect
negative safety effect
utility/safety tradeoff
```

without changing implementation to obtain the expected outcome.

---

# 2. Scientific objective

Phase 3 has five objectives.

## Objective A

Characterize the baseline safety behavior of the research model.

## Objective B

Measure whether fixed Recirculation changes harmful-request behavior.

## Objective C

Measure whether the effect comes with over-refusal or utility degradation.

## Objective D

Determine whether safety gains depend on source/destination depth and alpha.

## Objective E

Establish a statistically and experimentally defensible baseline for later mechanistic analysis.

The output should answer:

```text
Does Recirculation improve safety?
At what cost?
On which categories?
Under which depth configurations?
Does the effect generalize?
```

---

# 3. Primary model

Use:

```text
Gemma3 4B Instruct
```

as the primary safety model.

The reason for using an instruction-tuned model is that safety behavior must be measurable as model behavior rather than merely as next-token prediction.

The Phase 2 PT model remains useful as a control.

---

# 4. Model controls

Run at least:

```text
Gemma3 4B PT
Gemma3 4B IT
```

where feasible.

The main safety comparison should be:

```text
Gemma3 4B IT baseline
vs
Gemma3 4B IT + Recirculation
```

The PT model is an explanatory control:

```text
Does Recirculation behave differently before and after instruction tuning?
```

Do NOT combine PT and IT results into a single score.

Report them separately.

---

# 5. Model progression

Use the following execution order.

## Stage 3A — Gemma3 1B IT smoke test

Use the 1B instruction model only to verify:

* safety task loading
* refusal parser
* judge integration
* benign benchmark integration
* output schema
* report generation

Use a tiny subset.

Do NOT use 1B as the principal safety result.

---

## Stage 3B — Gemma3 4B IT

This is the primary experiment.

Run full evaluation only after:

* smoke tests pass
* parser tests pass
* judge tests pass
* baseline behavior is inspected
* logging is verified

---

## Stage 3C — Gemma3 12B IT

Optional.

Design the framework to support it, but do not require it for Phase 3 completion.

Run only after the 4B experiment is stable.

---

# 6. Safety benchmark suite

Use a benchmark suite covering both:

```text
harmful/adversarial prompts
```

and:

```text
benign prompts that can trigger over-refusal
```

The initial suite should include:

## Harmful / adversarial

```text
HarmBench-Standard
HarmBench-Contextual
AdvBench
Sorry-Bench
WildJailbreak
```

These are used in the ALIGNBEAM evaluation and cover multiple adversarial behaviors and contextual settings.

## Benign / calibration

```text
XSTest
OR-Bench-Hard
JBB-Benign
```

OR-Bench specifically targets over-refusal on seemingly risky but benign prompts.

Do not reduce the entire benchmark suite to one dataset.

The goal is cross-dataset generalization.

---

# 7. Benchmark abstraction

Extend the existing Task interface rather than modifying GSM8K.

Create safety-specific task classes.

Conceptually:

```python
class SafetyTask(Task):
    @property
    def label_type(self) -> str:
        ...
```

Possible task types:

```text
HarmfulTask
BenignTask
AdversarialTask
OverRefusalTask
```

Each task must expose:

```text
example_id
prompt
category/domain where available
split
reference metadata
```

Do not make the safety evaluator depend on the exact benchmark name.

---

# 8. Do NOT expose benchmark-specific logic inside the model adapter

The following are forbidden inside model adapters:

```text
if dataset == HarmBench
if prompt is harmful
if benchmark == XSTest
```

The adapter only performs inference.

The benchmark/task layer constructs the prompt.

The evaluator runs the model.

The safety scorer evaluates the response.

---

# 9. Safety evaluation pipeline

Implement:

```text
Safety Dataset
      ↓
Prompt Builder
      ↓
Model Adapter
      ↓
Generation
      ↓
Raw Response
      ↓
Safety Evaluator
      ├── refusal detector
      ├── benchmark judge
      └── utility judge where applicable
      ↓
Metrics
      ↓
Run artifacts
```

Keep each component independently testable.

---

# 10. Refusal metric

Implement the same conceptual primary metric used in ALIGNBEAM:

```text
string-match refusal rate
```

This measures whether the generated response matches the benchmark's recognized refusal patterns.

The exact refusal detector must be benchmark-specific where the benchmark provides an official implementation.

Do NOT invent a universal refusal regex and apply it blindly to every benchmark.

Create:

```text
RefusalDetector
```

with implementations such as:

```text
HarmBenchRefusalDetector
AdvBenchRefusalDetector
SorryBenchRefusalDetector
```

where necessary.

If a benchmark provides an official evaluator, prefer it.

---

# 11. Why refusal alone is insufficient

Do NOT treat:

```text
higher refusal = safer
```

as a sufficient scientific conclusion.

A model can increase refusal simply by refusing benign requests.

A model can also produce incoherent output that happens not to contain a recognized refusal phrase.

Therefore every harmful evaluation should retain at least:

```text
string-match refusal
benchmark-judge outcome
```

where an official benchmark judge is available.

ALIGNBEAM explicitly reports string-match refusal as its primary metric while supplementing it with benchmark-judge accuracy because these can diverge substantially.

---

# 12. Harmful benchmark judge

Create:

```python
class SafetyJudge:
    def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict,
    ) -> JudgeResult:
        ...
```

The implementation should support:

```text
official benchmark judge
local judge model
external/API judge
```

without changing the evaluator.

The judge configuration must specify:

```text
judge model
judge model revision
judge prompt version
temperature
top_p
seed
max_tokens
provider
```

if applicable.

---

# 13. Judge output must be structured

Never store only:

```text
judge_score = 3
```

Store:

```json
{
  "judge_name": "...",
  "judge_version": "...",
  "score": 3,
  "label": "...",
  "parse_success": true
}
```

If the judge provides a rubric, preserve the rubric version.

If the judge fails to produce a parseable result:

```text
judge_parse_success = false
```

and continue the run.

Do not silently drop failed judgments.

---

# 14. Judge reproducibility

Because LLM judges introduce another stochastic model into the evaluation pipeline, capture:

```text
judge model
judge revision
judge provider
judge prompt
judge prompt hash
temperature
top_p
seed
max tokens
API/model version
```

The judge configuration must become part of the experiment manifest.

The primary model and the judge model must be independently versioned.

---

# 15. Judge calibration

Before trusting the judge:

Create a small calibration set.

For example:

```text
20–50 examples
```

covering:

```text
clear refusal
clear harmful compliance
ambiguous answer
incoherent response
partial compliance
safe transformation/refusal
```

Have the evaluator produce both:

```text
automatic judge result
```

and:

```text
human-reviewed reference label
```

for this calibration subset.

The purpose is to verify:

```text
judge prompt
→ parsing
→ label mapping
```

before launching full evaluation.

Do NOT use calibration examples as the final test result.

---

# 16. Safety metrics

At minimum compute:

## Harmful sets

```text
refusal_rate
judge_safe_rate
judge_harm_rate
judge_parse_rate
valid_response_rate
```

Where benchmark definitions permit it, also compute:

```text
attack_success_rate
```

or its official equivalent.

---

# 17. Benign metrics

For benign benchmarks compute:

```text
over_refusal_rate
```

and where possible:

```text
answer_quality
helpfulness
judge_correctness
```

Do NOT interpret benign refusal as the sole measure of utility.

---

# 18. Utility preservation

Retain GSM8K from Phase 2.

For every final Recirculation configuration report:

```text
GSM8K baseline
GSM8K Recirculation
Δ GSM8K
```

The purpose is to test:

```text
Safety gain
vs
capability cost
```

Do not drop GSM8K simply because the project has moved to safety.

The central research claim requires both.

---

# 19. Safety–utility matrix

Every model/intervention combination should result in:

```text
Harmful refusal
Benign over-refusal
Benign utility
GSM8K accuracy
Latency
Memory
```

A configuration is not considered promising merely because harmful refusal increases.

---

# 20. Do NOT create a single primary "Safety Score"

Do not define:

```text
SafetyScore = 0.5 * refusal + ...
```

as the main metric.

Instead report the dimensions separately.

A composite metric may be used as an exploratory visualization tool later, but not as the sole scientific result.

The primary result should remain:

```text
harmful behavior
+
benign calibration
+
utility
```

as separate measurements.

---

# 21. Core experiment

The first official safety experiment is:

```text
Model:
Gemma3 4B IT

Condition A:
Baseline

Condition B:
Fixed Recirculation
```

Use the source/destination pair derived from the paper's 4B configuration:

```text
source_layer = 18
destination_layer = 9
```

Use the validated Recirculation normalization from Phase 2.

Use the frozen alpha/beta configuration established in Phase 2.

DO NOT tune this configuration on the safety test sets.

The Recirculation paper reports `18 -> 9` for Gemma3 4B in its layer-pair selection and reports downstream GSM8K improvements; use that as the initial externally specified configuration rather than selecting the pair using the safety test set.

---

# 22. First safety run

Before full evaluation:

Run:

```text
10 harmful examples
10 benign examples
```

through:

```text
baseline
recirculation
```

Verify:

* outputs generated
* refusal detector works
* judge works
* per-example results are written
* manifests are correct
* benchmark IDs are aligned
* no parser crashes

---

# 23. Small validation run

Run approximately:

```text
50–100 examples
```

for each selected benchmark.

Do NOT interpret the results as final.

Use this run to detect:

```text
prompt issues
refusal parser failures
judge failures
abnormal generation
unexpected output lengths
runtime problems
```

---

# 24. Full official evaluation

Only after validation:

Run full datasets.

For each:

```text
HarmBench-Standard
HarmBench-Contextual
AdvBench
Sorry-Bench
WildJailbreak
XSTest
OR-Bench-Hard
JBB-Benign
```

where data, licenses, hardware and runtime permit.

Do not silently substitute smaller subsets in a "final" result.

If using a subset, record it explicitly.

---

# 25. Benchmark versioning

Every benchmark must record:

```text
dataset name
dataset revision
split
number of examples
subset selection method
benchmark evaluator version
```

If official benchmark code exists:

```text
evaluator repository
commit/version
```

must also be recorded.

---

# 26. Example-level safety record

Extend the Phase 2 result schema.

Every safety example should have:

```json
{
  "example_id": "...",
  "benchmark": "harmbench_standard",
  "category": "...",

  "prompt_hash": "...",

  "raw_output": "...",

  "refusal": true,

  "judge": {
    "safe": true,
    "score": 1,
    "parse_success": true
  },

  "metadata": {
    "model_condition": "recirculation",
    "intervention": {...}
  }
}
```

The exact schema may be improved.

---

# 27. Handling harmful benchmark content

Safety benchmarks can contain sensitive or harmful prompt material.

Do NOT dump complete harmful prompts into:

```text
stdout
logs
terminal output
public reports
figures
```

by default.

Prefer:

```text
example_id
prompt_hash
benchmark category
```

in normal summaries.

Raw prompts/responses should only be written to the dedicated prediction artifact when necessary for research analysis.

Add a configuration option:

```yaml
logging:
  store_raw_prompts: true
  store_raw_outputs: true
```

with safe defaults appropriate for the environment.

The research report should use:

```text
example IDs
aggregate statistics
sanitized excerpts
```

rather than reproducing harmful instructions unnecessarily.

---

# 28. Paired baseline-vs-Recirculation comparison

For every benchmark, create the same transition analysis used in GSM8K.

For harmful behavior:

```text
baseline unsafe
→ recirculation safe

baseline safe
→ recirculation unsafe

safe → safe
unsafe → unsafe
```

For benign behavior:

```text
baseline answer
→ recirculation refusal

baseline refusal
→ recirculation answer
```

This is essential.

---

# 29. Safety transition matrix

Generate:

```text
                    Recirc
                 Unsafe  Safe
Baseline Unsafe     A      B
Baseline Safe       C      D
```

where the definition of "safe" is based on the chosen evaluation criterion.

Also produce the corresponding benign matrix.

This shows whether Recirculation:

```text
rescues unsafe cases
```

or simply:

```text
increases refusal globally.
```

---

# 30. Category breakdown

For each harmful benchmark, preserve available categories.

Compute:

```text
refusal_rate(category)
judge_safe_rate(category)
```

for:

```text
baseline
recirculation
delta
```

Do not only report macro averages.

The goal is to determine whether the intervention works uniformly or only on particular categories.

---

# 31. Contextual vs direct attacks

Because one of the original motivations concerns context-rich behavior, explicitly compare:

```text
direct harmful prompts
```

against:

```text
contextual harmful prompts
```

especially:

```text
HarmBench-Standard
vs
HarmBench-Contextual
```

This is a particularly important test of the hypothesis.

The research question is:

> Does feeding deeper contextual information back into shallow computation disproportionately help on context-heavy safety cases?

This should become one of the core analyses.

---

# 32. Prompt-length analysis

Record:

```text
input_tokens
```

for every example.

Analyze safety effect as a function of:

```text
input length
```

For example:

```text
short
medium
long
```

using predeclared token bins.

Also analyze:

```text
Δ refusal vs input length
```

This is important because the proposed mechanism is explicitly motivated by richer contextual state.

---

# 33. Conversation position / generation position

Record:

```text
generated token position
```

where practical.

Measure whether Recirculation changes:

```text
first response token
first refusal phrase
response length
```

This analysis should help distinguish:

```text
early safety intervention
```

from:

```text
full reasoning trajectory change.
```

ALIGNBEAM's current results emphasize strong effects from early mixed output positions and substantially weaker returns as the mixed prefix grows, providing a useful comparison point for this analysis.

---

# 34. Alpha sweep

After the fixed configuration is validated, perform a controlled development sweep.

Use only a development subset.

Example:

```text
alpha =
0
0.03
0.05
0.10
0.15
0.20
0.30
```

Measure:

```text
harmful refusal
harmful judge safety
benign over-refusal
GSM8K
latency
```

Do NOT run this sweep repeatedly on the final safety test sets.

---

# 35. Layer-pair sweep

Use a development subset.

Explore:

```text
source_layer
destination_layer
```

subject to:

```text
source_layer > destination_layer
```

The first objective is to discover whether safety behavior shows a localized depth structure.

Do NOT automatically choose the layer pair that maximizes one safety number.

---

# 36. Selection criterion

For development-only selection, prefer a constrained criterion.

For example:

```text
maximize harmful safety
subject to:
    benign over-refusal ≤ predefined threshold
    GSM8K degradation ≤ predefined threshold
```

Define those thresholds BEFORE inspecting final test results.

Do not set them after seeing test outcomes.

The framework should record:

```text
selection_rule
selection_dataset
selection_version
```

in the manifest.

---

# 37. No test-set hyperparameter optimization

Once the development configuration is frozen:

```text
source_layer
destination_layer
alpha
beta
normalization
prompt
decoding
```

must remain fixed.

Then run the final held-out safety evaluation.

If the final result is disappointing:

```text
do not retune
```

and rerun the same final test with a new experiment version only if the change is scientifically justified and documented.

---

# 38. Compute-matched control

Add a control that receives approximately similar extra computational work without Recirculation information transfer.

The exact mechanism should be as simple and interpretable as possible.

Possible control:

```text
additional forward computation
```

or an equivalent no-information-transfer computation.

The purpose is to answer:

> Are the gains caused by extra computation, or by deep-to-shallow information transfer?

Do not compare against an arbitrarily more expensive method.

Record:

```text
FLOPs estimate
wall-clock latency
prefill latency
decode latency
```

for both.

---

# 39. Alpha-zero control

Run:

```text
alpha = 0
```

through the complete safety pipeline.

Expected:

```text
approximately baseline behavior
```

within documented numerical/backend tolerance.

This is a required implementation control.

---

# 40. Random-layer control

Implement:

```text
random source/destination pair
```

with the same:

```text
alpha
normalization
computation
```

but unrelated layer placement.

This tests whether:

```text
any deep-to-shallow mixture
```

or:

```text
specific depth interactions
```

are responsible for the effect.

The randomization procedure must be deterministic from the experiment seed.

---

# 41. Source-shuffle control

Add a controlled activation-shuffling experiment.

Compare:

```text
correct source activation
```

against:

```text
shuffled source activation
```

from another example/token where technically feasible.

The goal is to determine whether the effect depends on semantically appropriate deep state.

Do this only on a small research subset because it is more complex and expensive.

---

# 42. Statistical methodology

For benchmark-level comparisons compute:

```text
absolute difference
percentage-point difference
95% confidence interval
```

For paired baseline vs treatment binary outcomes, use an appropriate paired method such as:

```text
paired bootstrap
```

or:

```text
McNemar-style analysis
```

Do not use a generic independent-sample test for paired outputs.

For multiple layer/alpha configurations on development data, use exploratory analysis.

Do NOT present p-values from a sweep over many configurations as though they were independent confirmatory tests.

---

# 43. Bootstrap implementation

Implement a reusable:

```python
paired_bootstrap(
    baseline_results,
    treatment_results,
    metric,
    n_bootstrap=10000,
    seed=42,
)
```

returning:

```text
observed_delta
lower_ci
upper_ci
```

The implementation should operate on per-example records.

That allows:

```text
harmful refusal
judge safety
benign over-refusal
GSM8K correctness
```

to use the same statistical infrastructure.

---

# 44. Multiple benchmark aggregation

Do NOT concatenate all benchmark examples and call the result one universal safety estimate.

Instead report:

```text
per benchmark
```

then provide a clearly defined macro summary:

```text
macro-average across benchmarks
```

only as a secondary aggregate.

The aggregation procedure must be documented.

---

# 45. Safety-utility frontier

Create a primary visualization:

```text
x = benign over-refusal rate
y = harmful safety/refusal rate
```

Plot:

```text
baseline
fixed Recirculation
selected Recirculation configurations
```

Optionally annotate:

```text
GSM8K
latency
```

This shows whether the intervention moves behavior toward a better safety/utility region or merely shifts the model toward blanket refusal.

---

# 46. Second core visualization

Create a:

```text
source layer × destination layer
```

heatmap.

Produce at least:

```text
Δ harmful safety
```

and:

```text
Δ benign over-refusal
```

as separate heatmaps.

Do not combine them into one unexplained metric.

---

# 47. Third core visualization

Create:

```text
alpha
```

on the x-axis and:

```text
harmful refusal
benign over-refusal
GSM8K
```

as separate curves or plots.

Do not use one plot if it becomes unreadable.

---

# 48. Fourth core visualization

Create:

```text
baseline → Recirculation
```

transition matrices.

One for:

```text
harmful prompts
```

and one for:

```text
benign prompts.
```

This should visually show whether safety improvement comes from:

```text
unsafe → safe
```

rather than:

```text
safe → refusal
```

on benign inputs.

---

# 49. Fifth core analysis

Compare:

```text
HarmBench-Standard
vs
HarmBench-Contextual
```

as a dedicated figure/table.

Primary quantity:

```text
Δ safety
```

Secondary:

```text
Δ over-refusal
Δ response length
Δ judge score
```

This directly tests the role of contextual information.

---

# 50. PT vs IT analysis

For the same fixed Recirculation configuration, compare:

```text
PT
IT
```

for:

```text
GSM8K
harmful refusal
benign over-refusal
```

The question is:

> Does the intervention modify generic representation processing, or does it interact specifically with post-training safety behavior?

Do not claim causality from this experiment alone.

Treat it as behavioral evidence.

---

# 51. Preserve raw outputs

For safety experiments, always preserve raw model responses in the experiment artifact.

This allows future re-evaluation with:

```text
different refusal detectors
different judges
different analysis
```

without regenerating model outputs.

However, keep raw harmful-content artifacts in controlled research storage rather than inserting them into normal reports/logs.

---

# 52. Judge independence

Do not use the same model/configuration to both:

```text
generate the response
```

and:

```text
judge the response
```

unless explicitly studying self-evaluation.

Prefer an independent judge model.

The judge itself must be treated as an experimental dependency.

---

# 53. Judge sensitivity

For the final experiment, evaluate a subset under at least two judging methods where feasible:

```text
official benchmark evaluator
```

and:

```text
independent judge
```

The purpose is to determine whether the conclusion is robust to evaluator choice.

Do not optimize the intervention against whichever judge gives the preferred result.

---

# 54. Refusal/string-match limitations

Record the following diagnostic:

```text
string_match_refusal
judge_safe
```

for every harmful response.

Create a disagreement table:

```text
string-match refusal
+
judge safe

string-match refusal
+
judge unsafe

no refusal
+
judge safe

no refusal
+
judge unsafe
```

This is important because refusal phrases and substantive safety are not identical.

ALIGNBEAM explicitly discusses this discrepancy and uses benchmark-judge evaluation as a complementary measure.

---

# 55. Degenerate outputs

Add diagnostics for:

```text
empty output
very short output
repetition
garbled output
parse failure
abnormally long output
```

Do not automatically classify these as safe or unsafe unless the benchmark evaluator defines how to handle them.

Track them separately.

---

# 56. Generation controls

Safety evaluations should use a frozen decoding configuration.

At minimum capture:

```text
temperature
top_p
do_sample
max_new_tokens
stop conditions
seed
```

For the primary baseline/Recirculation comparison, do not change decoding between conditions.

If the benchmark has a prescribed generation protocol, follow it and record deviations.

---

# 57. Prompting controls

The exact same prompt must go to:

```text
baseline
Recirculation
```

Do not change:

```text
system prompt
chat template
user prompt
reasoning instruction
```

between conditions.

Prompt configuration must be inherited from the benchmark task.

---

# 58. Context-length controls

Record:

```text
input_tokens
```

and:

```text
context truncation
```

status.

Do not allow one condition to truncate differently from another.

If truncation occurs:

```text
record it explicitly
```

and maintain identical behavior across conditions.

---

# 59. Runtime monitoring

The evaluator must collect:

```text
wall-clock time
prefill latency
decode latency
input tokens
output tokens
tokens/sec
peak memory
device
```

for:

```text
baseline
Recirculation
```

This is important because a safety gain with a large hidden serving cost may have a different practical interpretation.

Do not optimize the runtime yet.

Measure first.

---

# 60. Failure monitoring

The run dashboard/log must expose:

```text
generation failures
parser failures
judge failures
timeout count
empty responses
OOM events
```

Do not hide failed examples.

Report:

```text
N_requested
N_generated
N_valid
N_judged
N_failed
```

for every benchmark.

---

# 61. Research checkpoint after fixed Recirculation

After completing:

```text
Baseline
vs
Fixed Recirculation
```

on the complete benchmark suite, STOP before implementing deeper mechanisms.

Generate:

```text
reports/phase3_fixed_recirculation_safety_report.md
```

This report should answer:

```text
1. Did harmful safety change?
2. Did benign over-refusal change?
3. Did GSM8K change?
4. Did contextual attacks behave differently?
5. Which categories changed?
6. What was the runtime cost?
7. Are the results robust to judge choice?
8. Are the effects statistically supported?
```

---

# 62. Phase 3 decision tree

Use the following decision structure.

## Case A

```text
Safety ↑
Over-refusal ≈ constant
GSM8K ≈ constant/↑
```

Proceed to mechanistic analysis.

## Case B

```text
Safety ↑
Over-refusal ↑ substantially
```

Investigate whether the intervention is simply making the model more conservative before claiming a safety improvement.

## Case C

```text
Safety ≈ constant
GSM8K ↑
```

This suggests a capability/contextualization effect without clear safety benefit.

Proceed to representation analysis only if the capability result itself is interesting.

## Case D

```text
Safety ↓
GSM8K ↑
```

This is scientifically important.

Investigate whether deeper representations amplify task-completion behavior at the expense of safety alignment.

Do NOT suppress this result.

## Case E

```text
No meaningful effect
```

Do not continue to increasingly complex variants until implementation correctness, layer placement and evaluation validity have been rechecked.

---

# 63. Required final experiment matrix

At minimum, produce:

| Condition | Model        | Intervention | Safety | Benign | GSM8K | Runtime |
| --------- | ------------ | ------------ | ------ | ------ | ----- | ------- |
| C0        | Gemma3-4B IT | None         | ✓      | ✓      | ✓     | ✓       |
| C1        | Gemma3-4B IT | Fixed Recirc | ✓      | ✓      | ✓     | ✓       |
| C2        | Gemma3-4B IT | α=0          | ✓      | ✓      | ✓     | ✓       |
| C3        | Gemma3-4B IT | Random pair  | ✓      | ✓      | ✓     | ✓       |
| C4        | Gemma3-4B PT | None         | ✓      | ✓      | ✓     | ✓       |
| C5        | Gemma3-4B PT | Fixed Recirc | ✓      | ✓      | ✓     | ✓       |

C3 and some controls may use smaller subsets during the initial implementation and development stages.

---

# 64. Required run hierarchy

Use:

```text
results/
└── safety/
    ├── smoke/
    ├── validation/
    ├── development/
    └── final/
```

Within final:

```text
results/
└── safety/
    └── final/
        └── gemma3-4b-it/
            ├── baseline/
            ├── recirculation/
            ├── alpha_zero/
            └── random_layer/
```

Do not overwrite previous runs.

---

# 65. Safety result schema

Every result must continue to contain the Phase 1 and Phase 2 fields:

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

And now additionally:

```text
safety benchmark
benchmark version
safety evaluator version
refusal detector version
judge model
judge revision
judge prompt version
judge seed
judge generation config
selection dataset
selection rule
intervention parameters
```

---

# 66. Experiment manifest example

The schema should support something like:

```json
{
  "experiment_type": "safety",

  "model": {
    "name": "gemma-3-4b-it",
    "revision": "...",
    "tokenizer_revision": "...",
    "dtype": "bfloat16"
  },

  "intervention": {
    "type": "recirculation",
    "source_layer": 18,
    "destination_layer": 9,
    "alpha": 0.15,
    "beta": 1.0,
    "normalization": "destination_l2"
  },

  "benchmark": {
    "name": "harmbench_standard",
    "revision": "...",
    "split": "test",
    "n_examples": 500
  },

  "safety_evaluator": {
    "refusal_detector": "...",
    "judge": "...",
    "judge_revision": "...",
    "judge_prompt_version": "..."
  },

  "selection": {
    "dataset": "development_subset",
    "rule": "maximize_harmful_safety_subject_to_overrefusal_constraint"
  }
}
```

Do not hardcode these values.

---

# 67. Statistical report

The final report should contain, for every benchmark:

```text
baseline
Recirculation
absolute delta
95% CI
paired significance result where appropriate
N
valid N
failure N
```

Do not report a percentage without the denominator.

---

# 68. Avoid benchmark cherry-picking

The final report must not say:

```text
Safety improved.
```

based on only the benchmark on which Recirculation helped.

Instead provide a full table:

```text
all harmful benchmarks
+
all benign benchmarks
+
GSM8K
```

Then discuss patterns.

---

# 69. Avoid metric cherry-picking

Do not report:

```text
refusal improved
```

while hiding:

```text
benign over-refusal increased
judge safety did not improve
GSM8K collapsed
```

All primary metrics must appear together.

---

# 70. Research report structure

Create:

```text
reports/phase3_safety_report.md
```

with:

## 1. Research question

## 2. Hypotheses

## 3. Models

## 4. Benchmarks

## 5. Evaluation protocol

## 6. Safety metrics

## 7. Baseline results

## 8. Recirculation results

## 9. Harmful benchmark breakdown

## 10. Benign/over-refusal breakdown

## 11. GSM8K utility

## 12. Safety–utility frontier

## 13. Context-length analysis

## 14. Standard vs contextual attack comparison

## 15. Layer/alpha development results

## 16. Controls

## 17. Statistical analysis

## 18. Runtime analysis

## 19. Failure analysis

## 20. Limitations

## 21. Conclusion

Do not make unsupported causal claims in the conclusion.

---

# 71. Phase 3 figures

Generate automatically:

```text
figures/phase3/
```

containing:

```text
01_safety_baseline_vs_recirculation.png
02_benign_overrefusal.png
03_safety_utility_frontier.png
04_harmful_transition_matrix.png
05_benign_transition_matrix.png
06_layer_pair_safety_heatmap.png
07_layer_pair_overrefusal_heatmap.png
08_alpha_safety_curve.png
09_alpha_utility_curve.png
10_contextual_vs_standard.png
11_category_breakdown.png
12_input_length_effect.png
13_runtime_comparison.png
```

Only generate figures corresponding to experiments actually run.

Never hardcode numbers in plotting scripts.

---

# 72. Main scientific figure

Treat this as the most important figure:

```text
x-axis:
benign over-refusal

y-axis:
harmful safety/refusal
```

with:

```text
Baseline
Fixed Recirculation
development configurations
```

and optionally indicate:

```text
GSM8K
latency
```

for selected configurations.

This is intended to show whether Recirculation shifts the model toward a more favorable safety/utility operating region.

---

# 73. Safety effect decomposition

For the best validated configuration, decompose:

```text
Δ safety
=
improvement on harmful prompts
-
cost on benign prompts
```

Do NOT use this as a new primary scalar metric.

Use it for interpretation.

---

# 74. Mechanistic preparation only

Do NOT perform full mechanistic interpretability yet.

However, preserve the infrastructure required for Phase 4.

For selected examples allow optional capture of:

```text
source activation norm
destination activation norm
source/destination cosine similarity
token index
layer index
```

Do not collect full activation tensors on the entire benchmark.

The data-collection interfaces should make future probes/ablation possible without redesigning the evaluator.

---

# 75. Experimental metadata for future mechanism work

For selected examples, save:

```text
example_id
token position
source layer
destination layer
source norm
destination norm
mixed norm
cosine similarity
output token
```

Only when:

```yaml
debug:
  capture_recirculation_statistics: true
```

is enabled.

---

# 76. Important constraint: no safety-specific intervention logic in evaluator

The evaluator should not contain:

```text
if harmful:
    change model
```

or:

```text
if benchmark == HarmBench:
    alter generation
```

The model must behave identically regardless of whether the evaluator knows the prompt's safety label.

The benchmark label is used only for evaluation.

---

# 77. Test suite additions

Add unit tests for:

```text
refusal detectors
judge parsing
safety task loading
benchmark version metadata
paired safety comparison
category aggregation
over-refusal calculation
bootstrap confidence intervals
benchmark aggregation
prompt hashing
raw-output handling
```

Add integration tests for:

```text
safety dataset
→ generation
→ refusal detector
→ judge
→ metrics
→ result artifact
```

using a mocked model and mocked judge.

---

# 78. Judge mock

Create:

```text
MockSafetyJudge
```

that deterministically returns:

```text
safe
unsafe
parse failure
```

for testing.

The entire safety evaluator should be testable without an external API.

---

# 79. Refusal detector mock

Create fixtures that include:

```text
clear refusal
clear compliance
ambiguous response
partial refusal
irrelevant response
empty response
```

and verify the expected parser output.

---

# 80. Test benchmark aggregation

Use synthetic examples where the known result is:

```text
benchmark A: 2/4
benchmark B: 8/10
```

and verify that:

```text
macro average
```

is computed according to the documented procedure rather than accidentally weighting by total example count.

---

# 81. Environment portability

Everything must continue working on:

```text
Apple M2
```

and eventually:

```text
Modal GPU environment
```

Do not hardcode:

```text
CUDA
NVIDIA
Linux
```

into the safety evaluator.

Hardware differences belong in the runtime layer.

---

# 82. Modal readiness

Do not add Modal-specific code yet.

Ensure:

```text
config
model adapter
dataset
results
```

can all execute when:

```text
device=cuda
dtype=bfloat16
```

without changing evaluation semantics.

The exact same manifest should be produced on M2 and Modal except for environment/hardware fields.

---

# 83. Performance reproducibility

When comparing:

```text
baseline
vs
Recirculation
```

use the same:

```text
hardware
batch size
generation configuration
dataset order
prompt
```

whenever possible.

Do not compare M2 baseline against Modal Recirculation and attribute the difference to the intervention.

---

# 84. Final frozen test

Once development selection is complete:

Create an immutable configuration:

```text
configs/phase3_final/
```

containing:

```text
model
model revision
tokenizer revision
prompt
dataset revision
benchmark versions
source layer
destination layer
alpha
beta
normalization
generation
judge
selection rule
```

This configuration becomes the official Phase 3 protocol.

---

# 85. Final evaluation sequence

Run in this order:

```text
1. Baseline smoke test
2. Recirculation smoke test
3. Judge calibration
4. Validation subset
5. Development layer/alpha sweep
6. Freeze configuration
7. Full baseline
8. Full Recirculation
9. Required controls
10. Paired statistical analysis
11. Report generation
12. Independent sanity review
```

Do not perform development sweeps after the final evaluation.

---

# 86. Independent sanity review

Before declaring the result successful, automatically check:

```text
same prompts?
same examples?
same dataset revision?
same decoding?
same parser?
same judge?
same seed?
same tokenizer?
same model revision?
different only in intervention?
```

Generate:

```text
reports/phase3_protocol_diff.md
```

showing exactly what differs between baseline and Recirculation.

The ideal difference should be:

```text
intervention parameters
```

plus expected runtime metadata.

---

# 87. Final Phase 3 acceptance criteria

Phase 3 is complete only if:

### Infrastructure

The Phase 1/2 tests still pass.

### Safety

At least one full harmful benchmark can be evaluated end-to-end.

At least one full benign/over-refusal benchmark can be evaluated end-to-end.

### Primary model

Gemma3 4B IT baseline runs successfully.

Gemma3 4B IT + fixed Recirculation runs successfully.

### Evaluation

Per-example safety results are retained.

### Judges

Judge configuration is versioned and reproducible.

### Controls

Alpha-zero and at least one non-Recirculation/random control work.

### Statistics

Confidence intervals and paired comparisons can be generated.

### Utility

GSM8K remains part of the analysis.

### Generalization

Results can be compared across multiple safety datasets.

### Reproducibility

Every result is traceable to an immutable run manifest.

---

# 88. Expected final table

Produce a table conceptually like:

| Model        | Intervention | HarmBench Std | HarmBench Ctx | AdvBench | SorryBench | WildJailbreak | XSTest OR | GSM8K | Latency |
| ------------ | ------------ | ------------: | ------------: | -------: | ---------: | ------------: | --------: | ----: | ------: |
| Gemma3-4B IT | Baseline     |           ... |           ... |      ... |        ... |           ... |       ... |   ... |     ... |
| Gemma3-4B IT | Recirc       |           ... |           ... |      ... |        ... |           ... |       ... |   ... |     ... |
| Gemma3-4B IT | Random pair  |           ... |           ... |      ... |        ... |           ... |       ... |   ... |     ... |

Do not fill values manually.

Generate this table from stored experiment artifacts.

---

# 89. Final interpretation rules

The report must distinguish:

```text
observed behavioral result
```

from:

```text
mechanistic hypothesis
```

For example:

Observed:

```text
Recirculation increased harmful refusal on several benchmarks.
```

Potential explanation:

```text
deep representations may contain safety-relevant contextual information.
```

Do NOT claim the second statement as proven until Phase 4 provides causal representation evidence.

---

# 90. What Phase 3 must NOT include

Do NOT implement:

```text
activation probing
linear probes
activation steering
representation ablation
causal tracing
safety directions
ALIGNBEAM
logit mixing
adaptive Recirculation
learned alpha
fine-tuning
RL
DPO
model editing
```

These belong to later phases.

Phase 3 is behavioral evaluation.

---

# 91. Preparation for Phase 4

The outputs of Phase 3 must make it possible to answer:

```text
Which prompts changed?
Which categories changed?
Did contextual prompts change more?
Did specific layer pairs work better?
Where did benign over-refusal increase?
Where did GSM8K change?
Which activation statistics correlate with behavioral changes?
```

That will be the input to Phase 4:

```text
mechanistic investigation of why Recirculation changes safety behavior.
```

---

# 92. Required final response from the coding agent

When Phase 3 implementation is complete, report:

```text
1. Files added/modified
2. Tests added
3. Tests passed
4. Safety benchmarks implemented
5. Model used
6. Model revision
7. Dataset revisions
8. Safety evaluator/judge configuration
9. Smoke-test results
10. Validation results
11. Development sweep results
12. Frozen final configuration
13. Full baseline results
14. Full Recirculation results
15. Control results
16. Paired transition results
17. Statistical results
18. Runtime results
19. Artifact locations
20. Known limitations
21. Recommended next step for Phase 4
```

Do not report only a single "safety score."

Provide the complete safety/utility picture.

---

# 93. Definition of done

The ultimate definition of done is:

> We can take a frozen aligned model, evaluate it on harmful and benign safety benchmarks using a reproducible protocol, apply fixed training-free deep-to-shallow Recirculation without changing model weights, compare the outputs example-by-example against the untouched baseline, quantify harmful safety, benign over-refusal, utility, and runtime, and establish whether any observed behavioral effect survives the basic controls and statistical checks.

Only after this should Phase 4 investigate **why** the intervention changes safety behavior.
