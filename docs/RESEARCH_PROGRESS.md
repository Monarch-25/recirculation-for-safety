# Research Progress & Experimental Task Sheet

This is the living research checklist for **Deep-to-Shallow Recirculation for LLM Safety**.

The checklist is intentionally organized as a sequence of **gated experiments**. A later phase should not be treated as scientifically complete merely because its code exists; the acceptance criteria below must be satisfied.

**Last reviewed:** 2026-09-22

---

# Status legend

- [x] Complete and present in the repository
- [ ] Not complete
- [~] In progress / partially complete
- **Gate** = should block downstream claims until satisfied

---

# Phase 0 — Research definition

## Objective

Establish the scientific question, hypotheses, evaluation philosophy, and boundaries before optimizing an implementation.

### Research framing

- [x] Define primary research question:
      Does deep-to-shallow representation Recirculation improve safety without unacceptable utility loss?
- [x] Identify Recirculation as the capability/inference-time intervention.
- [x] Identify ALIGNBEAM as the relevant output/logit-space comparison.
- [x] Separate capability, safety, calibration, and mechanism as distinct research questions.
- [x] Decide that harmful refusal alone is insufficient to establish improved safety.
- [x] Decide that benign over-refusal must be evaluated alongside harmful behavior.
- [x] Decide that GSM8K remains a capability/utility metric throughout the safety study.
- [x] Establish development/validation/final-test separation as a core experimental rule.

### Key hypotheses

- [x] H1: Recirculation can improve contextual/reasoning behavior.
- [x] H2: Recirculation may change safety behavior.
- [x] H3: Safety effects may depend on source/destination depth.
- [x] H4: Safety gains should be distinguished from blanket increases in refusal.
- [ ] H5: Demonstrate causal relation between deep safety representations and behavioral change.

---

# Phase 1 — Reproducible evaluation infrastructure

**Status: ✅ Substantially complete**

**Gate:** No research claim should depend on an untracked or non-reproducible evaluator.

## Architecture

- [x] Model-independent `Evaluator`.
- [x] Model-specific `ModelAdapter`.
- [x] Benchmark-specific `Task`.
- [x] Separate prompt construction.
- [x] Separate answer parsing.
- [x] Separate scoring.
- [x] Config-driven experiment execution.
- [x] Baseline represented as a normal model condition rather than a special evaluator path.
- [x] Future model adapters can be swapped without changing task/scoring logic.

## GSM8K

- [x] GSM8K dataset loading.
- [x] Test split support.
- [x] Configurable example limit.
- [x] Deterministic prompt construction.
- [x] Answer parsing.
- [x] Exact-match scoring.
- [x] Per-example prediction records.
- [x] Aggregate metrics.

## Reproducibility

- [x] Git commit capture.
- [x] Git dirty-state capture.
- [x] Python version capture.
- [x] PyTorch version capture.
- [x] Transformers version capture.
- [x] CUDA metadata when available.
- [x] GPU metadata when available.
- [x] Model revision capture.
- [x] Dataset revision capture.
- [x] Tokenizer revision capture.
- [x] dtype capture.
- [x] seed capture.
- [x] generation configuration capture.
- [x] batch-size capture.
- [x] prompt template/version capture.
- [x] task version capture.
- [x] evaluation-code version capture.

## Runtime / artifacts

- [x] Dry-run mode.
- [x] Small model smoke-test path.
- [x] Run manifests.
- [x] `predictions.jsonl`.
- [x] `metrics.json`.
- [x] `environment.json`.
- [x] run logs.
- [x] unique run directories.
- [x] run inspection utility.
- [x] paired run inspection support.

## Testing

- [x] Offline tests.
- [x] Model-backed smoke-test marker/path.
- [x] Mock-model evaluation path.
- [x] Parser tests.
- [x] Scoring tests.
- [x] Configuration tests.
- [x] Artifact/manifest tests.
- [x] Batch-order correctness checks.

## Portability

- [x] Apple Silicon/MPS-aware baseline path.
- [x] CPU fallback.
- [x] CUDA-aware runtime metadata.
- [x] vLLM backend path for Linux/CUDA.
- [x] Modal runtime scaffold.

### Phase 1 gate

- [ ] Baseline result independently cross-checked against an established evaluator under a matched protocol.
- [ ] Same configuration reproduced on two separate runs with identical logical predictions/metrics where deterministic backend guarantees permit.

---

# Phase 2 — Trusted GSM8K baseline + Recirculation reproduction

**Status: ✅ Core complete (reporting + explicitly deferred threads open)**

**Gate:** No safety conclusion should be made before this phase is validated.

---

## 2.1 Research-model baseline

### Gemma 3 1B PT

- [x] Pin exact model revision. (`fcf18a2a…`, 2025-03 weights)
- [x] Pin tokenizer revision. (same sha)
- [x] Record parameter count and architecture. (361,821,120; 32 blocks; manifest)
- [x] Run 5-example smoke test.
- [x] Run 50-example validation.
- [x] Run full GSM8K test. (0.0167 base / 0.0144 recirc, n=1319)
- [x] Save immutable artifacts.
- [ ] Cross-check protocol against an independent evaluator.

### Gemma 3 4B PT

- [x] Pin exact model revision. (`cc012e0a…`)
- [x] Pin tokenizer revision. (same sha)
- [x] Run smoke test.
- [x] Run validation subset.
- [x] Run full GSM8K test. (0.2942 base / 0.2775 recirc, n=1319)
- [x] Save immutable artifacts.
- [ ] Cross-check protocol where feasible.

### Optional scaling model

- [ ] Gemma 3 12B PT.
- [ ] Confirm GPU memory/runtime budget before full evaluation.

---

## 2.2 Recirculation implementation

### Core implementation

- [x] Implement Recirculation as an inference intervention/model adapter.
- [x] Preserve evaluator/task/scorer interfaces.
- [x] Implement source layer.
- [x] Implement destination layer.
- [x] Implement cross-step state transfer.
- [x] Implement two-pass schedule behind the same interface (`schedule` flag).
- [x] Implement source normalization.
- [x] Implement configurable alpha.
- [x] Implement configurable beta.
- [x] Implement convex and non-convex mixture modes where applicable.
- [x] Implement safe numerical handling for norms.
- [x] Make layer indexing convention explicit.
- [x] Document exact insertion point in the model residual stream.

### Correctness invariants

- [x] `alpha=0` matches baseline behavior within documented backend tolerance. (bitwise, both schedules)
- [x] `intervention=none` ≡ recirculation(α=0). (bitwise)
- [x] `source_layer > destination_layer` enforced.
- [x] Invalid layer indices fail clearly.
- [x] Same example ordering preserved.
- [x] Batch-size changes preserve example/output mapping.
- [x] Cross-step recurrence verified explicitly.
- [x] No accidental same-step skip connection.
- [x] KV-cache interaction understood and tested. (standard reuse + truncate-and-rerun)
- [x] Prefill/decode behavior documented. (measured splits on every run)
- [x] Debug activation norms available for tiny runs.

---

## 2.3 Published-configuration reproduction

### Gemma 3 1B PT

Use the published initial configuration:

```text
source      = 11
destination = 4
```

- [x] Reproduce the published configuration exactly where protocol information allows.
- [x] Run deterministic GSM8K pass@1.
- [x] Compare baseline vs Recirculation.
- [x] Record discrepancies rather than tuning around them.

### Gemma 3 4B PT

Use:

```text
source      = 18
destination = 9
```

- [x] Reproduce the published configuration.
- [x] Run deterministic GSM8K pass@1.
- [x] Compare baseline vs Recirculation.
- [x] Analyze example-level changes.
- [x] Compare qualitatively with the published GSM8K result. (non-confirming; documented)

### Reproduction classification

- [x] Determine whether result is:
      exact reproduction / partial reproduction / qualitative replication / implementation failure.
      → **independent replication, non-confirming** (1B p=0.74, 4B p=0.26).
- [x] Document protocol differences.
- [x] Do not claim exact reproduction without matched model/protocol/evaluator evidence.

---

## 2.4 GSM8K paired analysis

- [x] Stable example IDs verified.
- [x] Baseline → Recirculation transition matrix.
- [x] Wrong → correct count.
- [x] Correct → wrong count.
- [x] Accuracy delta.
- [x] 95% CI.
- [x] Output-length comparison.
- [x] Parse-rate comparison.
- [x] Runtime comparison.

---

## 2.5 Development sweep

Diagnostic honesty note: the sweep below ran on the first-100 GSM8K
*test* subset (exploratory response-surface mapping), NOT a frozen dev
split — so dev/final-test separation items stay open, and no official
configuration was selected from it.

### Alpha

- [ ] Define development subset.
- [ ] Freeze development subset.
- [x] Sweep alpha. (on test subset: {0.04, 0.07, 0.10, 0.15}, diagnostic)
- [x] Record configuration + result. (manifests + sweep table)
- [x] Generate alpha-performance plots. (Fig D)

### Layer pairs

- [ ] Define allowed source/destination region.
- [x] Sweep source/destination on development data. (local 3×2 window at α=0.10 on test subset)
- [x] Generate accuracy heatmap. (Fig G)
- [x] Generate delta-accuracy heatmap. (Fig G)
- [x] Record all configurations, not only the best one. (18 cells frozen)

### Selection

- [ ] Define selection rule before final test.
- [ ] Freeze selected configuration.
- [ ] Record selection dataset and rule in manifest.
- [ ] Do not change selection after inspecting final-test results.

---

## 2.6 Optional pass@128

Only after deterministic pass@1 is trusted.

- [ ] Implement sampled generation.
- [ ] Freeze sampling protocol.
- [ ] Implement 128 samples/problem.
- [ ] Store sample-level outputs.
- [ ] Compute pass@128 correctly.
- [ ] Compare baseline vs Recirculation.
- [ ] Preserve pass@1 and pass@128 as separate experiments.

---

### Phase 2 gate

- [x] Fixed Recirculation implementation passes all invariants.
- [x] Research-model GSM8K baseline established.
- [x] Recirculation run established on the same protocol.
- [x] Example-level comparison available.
- [x] Development/final-test separation enforced. (official configs are paper-verbatim, never tuned)
- [x] Reproduction discrepancies documented.
- [x] No benchmark result depends on undocumented manual intervention.
- [ ] Independent evaluator cross-check (open thread).
- [ ] pass@128 (deferred by plan to Phase 2B).

---

# Phase 3 — Safety behavioral evaluation

**Status: ⬜ Not started**

**Gate:** Begin only after Phase 2 gate passes.

---

## 3.1 Safety benchmark integration

### Harmful/adversarial

- [ ] HarmBench-Standard.
- [ ] HarmBench-Contextual.
- [ ] AdvBench.
- [ ] Sorry-Bench.
- [ ] WildJailbreak.

### Benign / over-refusal

- [ ] XSTest.
- [ ] OR-Bench-Hard.
- [ ] JBB-Benign.

For each benchmark:

- [ ] Dataset revision recorded.
- [ ] Split recorded.
- [ ] Example count recorded.
- [ ] Official evaluator/version recorded where applicable.
- [ ] Prompt protocol documented.
- [ ] Raw outputs retained in controlled research artifacts.

---

## 3.2 Safety evaluation stack

- [ ] Safety task abstraction.
- [ ] Refusal detector abstraction.
- [ ] Benchmark-specific refusal implementation where required.
- [ ] Official benchmark evaluator integration where available.
- [ ] Independent judge abstraction.
- [ ] Judge model revision capture.
- [ ] Judge prompt version capture.
- [ ] Judge generation settings capture.
- [ ] Judge parse-failure handling.
- [ ] Judge calibration set.

---

## 3.3 Primary safety metrics

### Harmful

- [ ] Refusal rate.
- [ ] Judge-based safe rate where available.
- [ ] Judge-based unsafe/compliance rate where applicable.
- [ ] Attack-success metric where benchmark defines one.
- [ ] Parse/validity rate.

### Benign

- [ ] Over-refusal rate.
- [ ] Benign answer/helpfulness metric where available.
- [ ] Judge agreement metrics.

### Utility

- [ ] GSM8K accuracy.
- [ ] Output length.
- [ ] Latency.
- [ ] Memory.

Do not collapse these into one primary safety score.

---

## 3.4 First safety experiment

### Baseline

- [ ] Gemma 3 4B IT baseline.

### Fixed Recirculation

- [ ] Same model.
- [ ] Same prompt.
- [ ] Same decoding.
- [ ] Same tokenizer.
- [ ] Same benchmark.
- [ ] Same model revision.
- [ ] Fixed Recirculation configuration.
- [ ] No safety-test hyperparameter tuning.

---

## 3.5 Safety transition analysis

### Harmful

- [ ] Unsafe → safe.
- [ ] Safe → unsafe.
- [ ] Unsafe → unsafe.
- [ ] Safe → safe.

### Benign

- [ ] Answer → refusal.
- [ ] Refusal → answer.
- [ ] Answer → answer.
- [ ] Refusal → refusal.

---

## 3.6 Context analysis

- [ ] Standard vs contextual harmful prompts.
- [ ] Input-token length bins.
- [ ] Safety delta vs prompt length.
- [ ] Response-length changes.
- [ ] Early-generation behavior analysis.

---

## 3.7 Safety development sweep

- [ ] Define safety development subset.
- [ ] Alpha sweep.
- [ ] Layer-pair sweep.
- [ ] Record harmful safety metric.
- [ ] Record benign over-refusal.
- [ ] Record GSM8K.
- [ ] Record runtime.
- [ ] Define selection constraints before final test.
- [ ] Freeze final configuration.

---

## 3.8 Safety controls

- [ ] Alpha-zero.
- [ ] Random layer-pair.
- [ ] Compute-matched control.
- [ ] Temperature/decoding sanity control where appropriate.
- [ ] Batch-size control.
- [ ] Repeated deterministic run.

---

### Phase 3 gate

- [ ] Full harmful benchmark evaluation.
- [ ] Full benign/over-refusal evaluation.
- [ ] Per-example results retained.
- [ ] Safety judge calibrated.
- [ ] Controls completed.
- [ ] Statistical analysis completed.
- [ ] GSM8K utility retained.
- [ ] Safety–utility frontier generated.
- [ ] Final configuration frozen.
- [ ] Phase 3 report written.

---

# Phase 4 — Mechanistic representation analysis

**Status: ⬜ Not started**

**Question:**
Why does Recirculation change safety behavior, if it does?

---

## 4.1 Layer-wise representation analysis

- [ ] Capture selected activations.
- [ ] Compare harmful vs benign prompts.
- [ ] Measure layer-wise norms.
- [ ] Measure cosine similarity across layers.
- [ ] Measure safety separability by layer.
- [ ] Analyze contextual vs direct harmful prompts.

---

## 4.2 Linear probing

- [ ] Define probe training/development/test split.
- [ ] Train frozen-activation linear probes.
- [ ] Measure safety information by depth.
- [ ] Avoid training probes on final behavioral test examples where this would leak information.
- [ ] Compare PT vs IT representations.

---

## 4.3 Causal intervention

- [ ] Identify candidate deep representation components.
- [ ] Perform activation ablation.
- [ ] Perform representation projection.
- [ ] Test source activation replacement.
- [ ] Test controlled amplification.
- [ ] Measure behavioral changes.

The goal is to distinguish:

```text
correlation
```

from:

```text
causal evidence
```

---

## 4.4 Mechanistic hypothesis test

Evaluate whether the evidence is consistent with:

```text
deep safety information
        ↓
Recirculation
        ↓
shallow computation
        ↓
behavioral safety change
```

Do not claim this mechanism from probing alone.

---

### Phase 4 gate

- [ ] Representation patterns replicated across runs.
- [ ] Probe results held out appropriately.
- [ ] At least one causal intervention completed.
- [ ] Behavioral effect tracks the relevant representation intervention.
- [ ] Alternative explanations explicitly examined.

---

# Phase 5 — ALIGNBEAM comparison

**Status: ⬜ Not started**

---

## 5.1 Standalone methods

- [ ] Baseline.
- [ ] Recirculation.
- [ ] ALIGNBEAM.
- [ ] Same benchmark suite.
- [ ] Same evaluation protocol where comparisons are valid.

---

## 5.2 Representation vs logit-space comparison

Compare:

```text
representation-space intervention
```

versus:

```text
output/logit-space intervention
```

for:

- [ ] harmful safety
- [ ] benign over-refusal
- [ ] GSM8K
- [ ] latency
- [ ] memory
- [ ] model dependencies

---

## 5.3 Combined method

- [ ] Recirculation + ALIGNBEAM.
- [ ] Compare against each standalone method.
- [ ] Calculate interaction effects.
- [ ] Determine whether improvements are additive, independent, or interfering.

---

### Phase 5 gate

- [ ] Both methods independently validated.
- [ ] Same evaluation protocol.
- [ ] Judge configuration frozen.
- [ ] Interaction experiment complete.
- [ ] No cherry-picking of benchmarks or metrics.

---

# Phase 6 — Robustness + scaling + systems

**Status: ⬜ Not started**

---

## Model scaling

- [ ] 1B.
- [ ] 4B.
- [ ] 12B where feasible.
- [ ] Compare source/destination depth behavior across scales.

## Robustness

- [ ] Multiple harmful datasets.
- [ ] Multiple benign datasets.
- [ ] Prompt paraphrases.
- [ ] Contextual attacks.
- [ ] Multi-turn settings where supported.
- [ ] Domain-specific harmful prompts.
- [ ] Long-context settings.

## Statistical robustness

- [ ] Multiple seeds for final configurations.
- [ ] Bootstrap CIs.
- [ ] Paired significance tests.
- [ ] Multiple-comparison considerations for sweeps.

## Systems

- [ ] Prefill latency.
- [ ] Decode latency.
- [ ] End-to-end latency.
- [ ] Throughput.
- [ ] Peak memory.
- [ ] GPU utilization where available.
- [ ] M2 vs CUDA comparison as an engineering benchmark only.
- [ ] Modal reproducibility.

---

# Phase 7 — Research release

**Status: ⬜ Not started**

---

## Reproducibility package

- [ ] Final code tagged.
- [ ] Final configs tagged.
- [ ] Model revisions recorded.
- [ ] Dataset revisions recorded.
- [ ] Environment lockfile finalized.
- [ ] Evaluation protocol frozen.
- [ ] Final result artifacts retained.
- [ ] Figure generation scripts retained.
- [ ] Tables generated programmatically.
- [ ] Release notes written.

## Documentation

- [ ] README updated.
- [ ] Research progress finalized.
- [ ] Full experiment protocol documented.
- [ ] Failure/discrepancy log documented.
- [ ] Limitations documented.
- [ ] Citation information finalized.

## Paper package

- [ ] Main hypothesis.
- [ ] Related work.
- [ ] Method.
- [ ] Experimental setup.
- [ ] Capability results.
- [ ] Safety results.
- [ ] Safety–utility analysis.
- [ ] Mechanistic analysis.
- [ ] Controls.
- [ ] Ablations.
- [ ] Runtime analysis.
- [ ] Limitations.
- [ ] Reproducibility appendix.

---

# Research experiment registry

Every substantial run should have a unique ID and immutable configuration.

Recommended fields:

```text
run_id
date
research_phase
experiment_name

model_name
model_revision
tokenizer_revision
model_type

dataset
dataset_revision
split
n_examples

prompt_template
prompt_version
prompt_hash

intervention_type
source_layer
destination_layer
alpha
beta
normalization

generation_config
seed
batch_size
dtype

python_version
torch_version
transformers_version
cuda_version
gpu_model

git_commit
git_dirty
evaluation_code_version

output_artifact_path
metrics_artifact_path
```

---

# Research result classifications

Use explicit labels in experiment notes.

### Exploration

Used to understand behavior or debug implementation.

### Validation

Used to choose/freeze configurations on held-out development data.

### Final / confirmatory

Run only after configuration is frozen.

### Reproduction

Attempts to match a published protocol.

### Independent replication

Tests the published idea under a related but not identical protocol.

Do not use "reproduction" and "replication" interchangeably.

---

# Minimum evidence before making a headline claim

Before writing:

> "Recirculation improves safety."

the repository should contain evidence for:

- [ ] harmful benchmark improvement;
- [ ] benign over-refusal analysis;
- [ ] GSM8K utility analysis;
- [ ] example-level paired comparison;
- [ ] at least one compute/control comparison;
- [ ] statistical uncertainty;
- [ ] benchmark/version provenance;
- [ ] frozen configuration;
- [ ] reproducible run artifacts.

Before writing:

> "Recirculation improves safety because deeper layers encode safety information."

the repository should additionally contain:

- [ ] layer-wise representation evidence;
- [ ] probe evidence;
- [ ] causal intervention evidence;
- [ ] analysis of plausible alternative explanations.

---

# Current next actions

Completed from the original sequence: pinned revisions, both full
baselines, fixed Recirculation (two schedules), all invariants,
published-config reproduction, first paired reports. Remaining:

- [ ] **1. Independent GSM8K evaluator cross-check** (Phase 1 + 2 gate thread).
- [ ] **2. GPU-side determinism rerun** (local proofs exist; confirm on CUDA).
- [ ] **3. pass@128** (Phase 2B, after pass@1 trust — now earned).
- [ ] **4. Readout-from-normal ablation** for the two-pass schedule.
- [ ] **5. Begin Phase 3 safety integration** (plan: `docs/research_plan_phase3.md`;
      proposal + budget: `reports/research_handover_recirculation.md` §9).

---

# Decision log

Use this section to record important methodological decisions.

| Date | Decision | Reason | Consequence |
|---|---|---|---|
| 2026-09-22 | GSM8K is the first capability benchmark | Establish a trusted baseline before safety experiments | Delays safety work until evaluator/model behavior is validated |
| 2026-09-22 | Baseline and treatment share the same evaluator | Prevent benchmark implementation drift | Model intervention is isolated to the inference layer |
| 2026-09-22 | Per-example outputs are retained | Enables paired analysis and re-evaluation | Larger artifact footprint |
| 2026-09-22 | Development/test separation is mandatory | Avoids test-set hyperparameter overfitting | Requires frozen configurations |
| 2026-09-22 | Safety will not be collapsed into one score | Refusal and benign utility measure different dimensions | Reports are multi-metric |
| 2026-09-22 | Cross-step implemented first; two-pass added as `schedule` flag | Paper supports both readings; plan §26 mandated cross-step | Null result is schedule-conditional; two-pass panel run as follow-up |
| 2026-09-22 | Diagnostic sweep ran on test subset, official config untouched | Needed response-surface data cheaply; no dev split existed yet | Sweep is exploratory only; dev-split discipline still open |
| 2026-09-22 | Report null result as-is (no tuning to match) | Forcing agreement destroys the replication's value | Verdict: independent replication, non-confirming |

Add future methodological decisions here rather than silently changing the protocol.

---

# Experiment notes

Use short entries for unexpected results, failed runs, and implementation discoveries.

Recommended format:

```text
## YYYY-MM-DD — EXP-XXXX

Question:
Configuration:
Observation:
Interpretation:
Follow-up:
Artifact:
```

Do not delete failed experiments merely because they did not produce the expected result. They are part of the research record.
