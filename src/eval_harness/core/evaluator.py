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
from pathlib import Path
from typing import Any

import yaml

from eval_harness.core.config import EvalConfig
from eval_harness.core.interfaces import ModelAdapter, Task
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
                "intervention": config.model.intervention,
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
            },
            "generation": config.generation.to_dict(),
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
        raw_outputs: list[str] = [""] * len(examples)
        batch_size = config.runtime.batch_size
        n_batches = (len(examples) + batch_size - 1) // batch_size
        for b, (start, batch_prompts) in enumerate(
            _chunked_with_start(prompts, batch_size)
        ):
            log.info("generation batch %d/%d (size %d)",
                     b + 1, n_batches, len(batch_prompts))
            outputs = model.generate(batch_prompts, config.generation)
            if len(outputs) != len(batch_prompts):
                raise RuntimeError(
                    f"Model adapter returned {len(outputs)} outputs for "
                    f"{len(batch_prompts)} prompts (batch {b})"
                )
            for j, out in enumerate(outputs):
                raw_outputs[start + j] = out

        # -- parse + score (per-example failures never crash the run) ---
        records: list[PredictionRecord] = []
        for i, (ex, prompt, raw) in enumerate(zip(examples, prompts, raw_outputs)):
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
        )

        # -- artifacts --------------------------------------------------
        io.write_json(run_dir / "manifest.json", manifest)
        io.write_text(run_dir / "config.yaml", config.to_yaml())
        io.write_json(run_dir / "metrics.json", metrics.to_dict())
        io.write_jsonl(run_dir / "predictions.jsonl",
                       [r.to_dict() for r in records])
        io.write_json(run_dir / "environment.json", environment)
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
