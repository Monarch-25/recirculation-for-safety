"""Model-independent evaluation orchestrator.

The evaluator owns orchestration only:

    dataset -> prompts -> batches -> model outputs -> parse -> score
    -> aggregate -> artifacts

It contains NO model-specific logic and NO benchmark-specific
conditionals. All benchmark semantics live in the ``Task``; all
generation internals live in the ``ModelAdapter``.
"""

from __future__ import annotations

import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from eval_harness.core.config import EvalConfig
from eval_harness.core.interfaces import BatchTiming, ModelAdapter, Task
from eval_harness.core.results import (
    RESULT_SCHEMA_VERSION,
    EvaluationResult,
    PredictionRecord,
    compute_metrics,
)
from eval_harness.runtime import reproducibility
from eval_harness.runtime.environment import collect_environment_metadata
from eval_harness.utils import hashing, io

log = logging.getLogger(__name__)


NULL_PERFORMANCE: dict[str, Any] = {
    "wall_clock_seconds": None,
    "prefill_seconds": None,
    "generation_seconds": None,
    "input_tokens": None,
    "output_tokens": None,
    "tokens_per_second": None,
    "peak_memory_bytes": None,
    "token_stats_complete": False,
}


def _reset_cuda_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak_bytes() -> int | None:
    """Peak device allocation; None where no CUDA API exists (MPS/CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


class _PerfTracker:
    """Accumulates BatchTiming across generate() calls and stages.

    Stage 0 writes per-example token counts; stage 1 (answer extraction)
    adds to them. Any gap — a missing report, misaligned lists, or a
    None entry — invalidates the totals (all-or-nothing, never partial).
    Timing sums accumulate independently per phase with their own
    completeness flags.
    """

    def __init__(self, n: int) -> None:
        self.in_tokens: list[int | None] = [None] * n
        self.out_tokens: list[int | None] = [None] * n
        self.prefill_sum = 0.0
        self.gen_sum = 0.0
        self.prefill_known = True
        self.gen_known = True
        self.tokens_complete = True

    def consume(self, model: Any, batch_prompts: list[str],
                start: int, stage: int, batch_idx: int) -> None:
        info = getattr(model, "last_batch_info", None)
        try:
            model.last_batch_info = None  # consume; avoid stale reuse
        except Exception:
            pass
        if not isinstance(info, BatchTiming):
            self.prefill_known = False
            self.gen_known = False
            self._invalidate()
            return
        if info.prefill_seconds is None:
            self.prefill_known = False
        else:
            self.prefill_sum += info.prefill_seconds
        if info.generation_seconds is None:
            self.gen_known = False
        else:
            self.gen_sum += info.generation_seconds
        batch_in = info.input_tokens or []
        batch_out = info.output_tokens or []
        if (len(batch_in) == len(batch_prompts)
                and len(batch_out) == len(batch_prompts)
                and all(isinstance(v, int) for v in batch_in + batch_out)):
            for j in range(len(batch_prompts)):
                i = start + j
                if stage == 0:
                    self.in_tokens[i] = batch_in[j]
                    self.out_tokens[i] = batch_out[j]
                elif (self.in_tokens[i] is None
                        or self.out_tokens[i] is None):
                    self._invalidate()
                    return
                else:
                    self.in_tokens[i] += batch_in[j]  # type: ignore[operator]
                    self.out_tokens[i] += batch_out[j]  # type: ignore[operator]
        else:
            self._invalidate()
            if batch_in or batch_out:
                log.warning(
                    "batch %d token lists misaligned "
                    "(got %d/%d for %d prompts); ignoring",
                    batch_idx, len(batch_in), len(batch_out),
                    len(batch_prompts))

    def _invalidate(self) -> None:
        self.tokens_complete = False
        self.in_tokens = [None] * len(self.in_tokens)
        self.out_tokens = [None] * len(self.out_tokens)

    def performance(self, wall_clock_seconds: float,
                    backend: str) -> dict[str, Any]:
        complete = (self.tokens_complete
                    and all(v is not None for v in self.in_tokens)
                    and all(v is not None for v in self.out_tokens))
        in_sum = sum(v for v in self.in_tokens if v is not None)
        out_sum = sum(v for v in self.out_tokens if v is not None)
        return {
            "wall_clock_seconds": wall_clock_seconds,
            "prefill_seconds": self.prefill_sum if self.prefill_known else None,
            "generation_seconds": self.gen_sum if self.gen_known else None,
            "input_tokens": in_sum if complete else None,
            "output_tokens": out_sum if complete else None,
            "tokens_per_second": (
                out_sum / self.gen_sum
                if self.gen_known and complete and self.gen_sum > 0
                else None),
            # CUDA peak is only meaningful for in-process execution. The
            # vLLM backend runs in a child EngineCore process, so the
            # parent counter stays 0 — record null instead of a lie.
            "peak_memory_bytes": (
                None if backend == "vllm" else _cuda_peak_bytes()),
            "token_stats_complete": complete,
        }


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _slugify_model(name: str) -> str:
    slug = name.split("/")[-1].strip().lower().replace("_", "-")
    return "".join(c if (c.isalnum() or c == "-") else "-" for c in slug).strip("-")


def _chunked(seq: list, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield i, seq[i:i + batch_size]


class Evaluator:
    """Orchestrates evaluation of a Task with a ModelAdapter."""

    def __init__(self, output_root: str | Path | None = None) -> None:
        self._output_root_override = Path(output_root) if output_root else None

    # -- run directory --------------------------------------------------
    def make_run_dir(self, config: EvalConfig, run_id: str) -> Path:
        root = self._output_root_override or Path(config.runtime.output_dir)
        run_dir = (
            root / config.task.name
            / _slugify_model(config.model.name)
            / f"run_{run_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    # -- manifest -------------------------------------------------------
    def build_manifest(
        self,
        *,
        config: EvalConfig,
        run_id: str,
        model: ModelAdapter,
        task: Task,
        prompt_text: str,
        prompt_hash: str,
        environment: dict[str, Any],
        git: dict[str, Any],
        command: str,
        timestamp: str,
        num_examples: int,
        dataset_revision: str | None,
        performance: dict[str, Any] | None = None,
        extraction_hash: str | None = None,
    ) -> dict[str, Any]:
        model_meta = model.get_metadata()
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment": {
                "name": config.experiment.name,
                "evaluation_code_version": config.experiment.evaluation_code_version,
                "command": command,
            },
            "git": git,
            "model": {
                "name": model_meta.get("name", config.model.name),
                "revision": model_meta.get("revision", config.model.revision),
                "tokenizer_revision": model_meta.get(
                    "tokenizer_revision", config.model.tokenizer_revision
                ),
                "dtype": model_meta.get("dtype", config.model.dtype),
                "device": model_meta.get("device", None),
                "device_requested": config.model.device,
                "intervention": config.model.intervention.to_dict(),
                "extra": {
                    k: v for k, v in model_meta.items()
                    if k not in ("name", "revision", "tokenizer_revision",
                                 "dtype", "device", "intervention_type")
                },
            },
            "task": {
                "name": task.name,
                "version": task.version,
                "split": config.task.split,
                "limit": config.task.limit,
                "num_examples": num_examples,
                "dataset_id": config.task.dataset_id,
                "dataset_config": config.task.dataset_config,
                "dataset_revision": dataset_revision,
            },
            "prompt": {
                "name": config.prompt.template_name,
                "version": config.prompt.template_version,
                "hash": prompt_hash,
                "canonical_template_preview": prompt_text[:2000],
                "extraction": (
                    None if (config.prompt.extraction_template_name is None
                             or extraction_hash is None)
                    else {
                        "name": config.prompt.extraction_template_name,
                        "version": config.prompt.extraction_template_version,
                        "hash": extraction_hash,
                    }),
            },
            "generation": config.generation.to_dict(),
            "performance": dict(performance) if performance else dict(NULL_PERFORMANCE),
            "runtime": {
                "python_version": environment.get("python_version"),
                "torch_version": environment.get("torch_version"),
                "transformers_version": environment.get("transformers_version"),
                "datasets_version": environment.get("datasets_version"),
                "cuda_version": environment.get("cuda_version"),
                "cuda_available": environment.get("cuda_available"),
                "mps_available": environment.get("mps_available"),
                "gpu_model": environment.get("gpu_model"),
                "gpu_count": environment.get("gpu_count"),
                "device": model_meta.get("device", environment.get("device")),
                "batch_size": config.runtime.batch_size,
                "seed": config.generation.seed,
                "os": environment.get("os"),
                "platform": environment.get("platform"),
                "machine": environment.get("machine"),
                "hostname": environment.get("hostname"),
                "cpu": environment.get("cpu"),
                "ram_gb": environment.get("ram_gb"),
                "timezone": environment.get("timezone"),
            },
        }

    # -- main entry point -----------------------------------------------
    def evaluate(
        self,
        *,
        task: Task,
        model: ModelAdapter,
        config: EvalConfig,
        run_id: str | None = None,
        command: str | None = None,
        log_file: Path | None = None,
    ) -> EvaluationResult:
        """Run full evaluation, write artifacts, return the result.

        Ordering guarantee: ``records[i]`` corresponds to the i-th
        loaded example regardless of ``batch_size``.
        """
        if config.runtime.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        reproducibility.seed_everything(config.generation.seed)

        run_id = run_id or reproducibility.new_run_id()
        timestamp = _utc_now().isoformat()
        command = command or " ".join(sys.argv)
        run_dir = self.make_run_dir(config, run_id)

        # Attach a file log handler for logs.txt (evaluator-owned).
        file_handler = None
        if log_file is None:
            log_file = run_dir / "logs.txt"
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        try:
            return self._run(
                task=task, model=model, config=config,
                run_id=run_id, timestamp=timestamp,
                command=command, run_dir=run_dir,
            )
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()

    def _run(self, *, task, model, config, run_id, timestamp, command, run_dir):
        log.info("run started: %s", run_id)
        log.info("resolved model: %s", model.model_id)
        log.info("resolved task: %s v%s", task.name, task.version)

        environment = collect_environment_metadata()
        git = reproducibility.get_git_metadata()
        if git.get("dirty"):
            log.warning("git working tree is dirty; recording git_dirty=true")

        # -- dataset ----------------------------------------------------
        examples = task.load(split=config.task.split, limit=config.task.limit)
        log.info("dataset loaded: %d examples", len(examples))
        if not examples:
            raise ValueError("Task returned zero examples")

        # -- prompts (deterministic, saved per example) -----------------
        prompts = [task.build_prompt(ex) for ex in examples]
        prompt_hash = hashing.sha256_hex(
            f"{config.prompt.template_name}\n"
            f"{config.prompt.template_version}\n"
            f"{prompts[0] if prompts else ''}"
        )

        # -- generation in batches, order-preserving --------------------
        # Timing/token stats (P2 §32) come from the optional per-call
        # BatchTiming adapters publish as `last_batch_info`. Anything an
        # adapter cannot measure stays null (never fabricated).
        _reset_cuda_peak()
        tracker = _PerfTracker(len(examples))
        gen_wall_start = time.perf_counter()
        batch_size = config.runtime.batch_size

        def _generate_batched(prompt_list: list[str], stage: int,
                              label: str) -> list[str]:
            outputs: list[str] = [""] * len(prompt_list)
            n_batches = (len(prompt_list) + batch_size - 1) // batch_size
            for b, (start, batch_prompts) in enumerate(
                _chunked_with_start(prompt_list, batch_size)
            ):
                log.info("%s batch %d/%d (size %d)", label,
                         b + 1, n_batches, len(batch_prompts))
                chunk = model.generate(batch_prompts, config.generation)
                if len(chunk) != len(batch_prompts):
                    raise RuntimeError(
                        f"Model adapter returned {len(chunk)} outputs for "
                        f"{len(batch_prompts)} prompts ({label} {b})"
                    )
                for j, out in enumerate(chunk):
                    outputs[start + j] = out
                tracker.consume(model, batch_prompts, start, stage, b)
            return outputs

        # Stage 1: reasoning (or the full response for single-stage).
        reasoning = _generate_batched(prompts, 0, "generation")

        # Stage 2 (optional): answer extraction. Tasks without
        # build_extraction_prompt, or returning all-None, stay
        # single-stage with identical behavior to previous versions.
        raw_outputs = reasoning
        reasoning_list: list[str | None] = [None] * len(examples)
        extraction_prompts: list[str] | None = None
        extraction_hash: str | None = None
        extract_fn = getattr(task, "build_extraction_prompt", None)
        if extract_fn is not None:
            maybe_prompts = [extract_fn(ex, z)
                             for ex, z in zip(examples, reasoning)]
            if any(p is None for p in maybe_prompts):
                if not all(p is None for p in maybe_prompts):
                    raise ValueError(
                        "build_extraction_prompt returned None for only "
                        "some examples; must be all or nothing")
                log.info("task declined extraction; single-stage run")
            else:
                extraction_prompts = maybe_prompts
                reasoning_list = list(reasoning)
                raw_outputs = _generate_batched(
                    extraction_prompts, 1, "extraction")
                extraction_hash = hashing.sha256_hex(
                    f"{config.prompt.extraction_template_name}\n"
                    f"{config.prompt.extraction_template_version}\n"
                    f"{extraction_prompts[0] if extraction_prompts else ''}"
                )
        gen_wall = time.perf_counter() - gen_wall_start
        performance = tracker.performance(gen_wall, config.model.backend)

        # -- parse + score (per-example failures never crash the run) ---
        # raw_outputs holds the SCORED text: stage-2 output when an
        # extraction stage ran, else the stage-1 response.
        records: list[PredictionRecord] = []
        ext_list = (extraction_prompts if extraction_prompts is not None
                    else [None] * len(examples))
        for i, (ex, prompt, raw, rsn, ext) in enumerate(
                zip(examples, prompts, raw_outputs, reasoning_list,
                    ext_list)):
            try:
                parsed = task.parse_answer(raw)
            except Exception as exc:  # parser must not kill the run
                log.warning("parse failed for %s: %s", ex.example_id, exc)
                from eval_harness.core.interfaces import ParsedAnswer
                parsed = ParsedAnswer(value=None, success=False)
            try:
                ref_parsed = task.parse_reference(ex)
            except Exception:
                from eval_harness.core.interfaces import ParsedAnswer
                ref_parsed = ParsedAnswer(value=None, success=False)
            try:
                s = float(task.score(parsed, ex))
                correct = s >= 0.5
            except Exception as exc:
                log.warning("score failed for %s: %s", ex.example_id, exc)
                s, correct = 0.0, False
            records.append(PredictionRecord(
                example_id=ex.example_id,
                index=i,
                prompt=prompt,
                raw_output=raw,
                parsed_answer=parsed.value,
                reference_answer=ex.reference_answer,
                parsed_reference=ref_parsed.value,
                parse_success=parsed.success,
                correct=correct,
                score=s,
                input_tokens=tracker.in_tokens[i],
                output_tokens=tracker.out_tokens[i],
                question=ex.question,
                reasoning=rsn,
                extraction_prompt=ext,
            ))

        metrics = compute_metrics(records)
        log.info("metrics computed: acc=%.4f parse=%.4f n=%d",
                 metrics.accuracy, metrics.parse_rate, metrics.num_examples)

        dataset_revision = _resolve_dataset_revision(task, config)
        manifest = self.build_manifest(
            config=config, run_id=run_id, model=model, task=task,
            prompt_text=prompts[0] if prompts else "",
            prompt_hash=prompt_hash, environment=environment, git=git,
            command=command, timestamp=timestamp,
            num_examples=len(examples), dataset_revision=dataset_revision,
            performance=performance,
            extraction_hash=extraction_hash,
        )

        # -- artifacts --------------------------------------------------
        io.write_json(run_dir / "manifest.json", manifest)
        io.write_text(run_dir / "config.yaml", config.to_yaml())
        io.write_json(run_dir / "metrics.json", metrics.to_dict())
        io.write_jsonl(run_dir / "predictions.jsonl",
                       [r.to_dict() for r in records])
        io.write_json(run_dir / "environment.json", environment)
        # Optional experiment tracking (never fails the run).
        from eval_harness.runtime import wandb_logging
        wandb_logging.log_evaluation(
            config, run_id=run_id, run_dir=str(run_dir),
            metrics=metrics, performance=manifest.get("performance"),
        )
        # logs.txt already streaming via handler; append run summary line.
        with open(run_dir / "logs.txt", "a") as f:
            f.write(f"run completed: {run_id} "
                    f"acc={metrics.accuracy:.4f} n={metrics.num_examples}\n")

        log.info("results written to %s", run_dir)
        log.info("run completed")
        return EvaluationResult(
            run_id=run_id, records=records, metrics=metrics,
            manifest=manifest, run_dir=str(run_dir),
        )


def _chunked_with_start(seq: list[str], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield i, seq[i:i + batch_size]


def _resolve_dataset_revision(task: Task, config: EvalConfig) -> str | None:
    rev = getattr(task, "dataset_revision", None)
    if callable(rev):
        try:
            return rev()
        except Exception:
            return config.task.dataset_revision
    if isinstance(rev, str):
        return rev
    return config.task.dataset_revision
