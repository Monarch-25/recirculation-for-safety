"""Typed configuration objects parsed from YAML.

The YAML config is user-facing; these dataclasses are the typed,
validated internal representation. Model implementations must NOT
depend on the YAML structure directly — they receive typed fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eval_harness.core.interfaces import GenerationConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str = "hf"  # hf | vllm
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "float32"
    device: str = "auto"  # auto | cpu | mps | cuda
    trust_remote_code: bool = False
    use_chat_template: bool = True
    attn_implementation: str | None = None  # hf backend only
    # vLLM backend only:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    max_model_len: int | None = None
    intervention_type: str = "none"
    intervention: dict[str, Any] = field(default_factory=lambda: {"type": "none"})


@dataclass(frozen=True)
class TaskConfig:
    name: str = "gsm8k"
    version: str = "1.0"
    split: str = "test"
    limit: int | None = None
    dataset_id: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_revision: str | None = None


@dataclass(frozen=True)
class PromptConfig:
    template_name: str = "gsm8k_cot_v1"
    template_version: str = "1.0"


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int = 1
    output_dir: str = "results"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    evaluation_code_version: str = "0.1.0"


@dataclass(frozen=True)
class EvalConfig:
    model: ModelConfig
    task: TaskConfig
    prompt: PromptConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    experiment: ExperimentConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.model.name,
                "backend": self.model.backend,
                "revision": self.model.revision,
                "tokenizer_revision": self.model.tokenizer_revision,
                "dtype": self.model.dtype,
                "device": self.model.device,
                "trust_remote_code": self.model.trust_remote_code,
                "use_chat_template": self.model.use_chat_template,
                "attn_implementation": self.model.attn_implementation,
                "tensor_parallel_size": self.model.tensor_parallel_size,
                "gpu_memory_utilization": self.model.gpu_memory_utilization,
                "enforce_eager": self.model.enforce_eager,
                "max_model_len": self.model.max_model_len,
                "intervention": self.model.intervention,
            },
            "task": {
                "name": self.task.name,
                "version": self.task.version,
                "split": self.task.split,
                "limit": self.task.limit,
                "dataset_id": self.task.dataset_id,
                "dataset_config": self.task.dataset_config,
                "dataset_revision": self.task.dataset_revision,
            },
            "prompt": {
                "template_name": self.prompt.template_name,
                "template_version": self.prompt.template_version,
            },
            "generation": self.generation.to_dict(),
            "runtime": {
                "batch_size": self.runtime.batch_size,
                "output_dir": self.runtime.output_dir,
                "seed": self.runtime.seed,
            },
            "experiment": {
                "name": self.experiment.name,
                "evaluation_code_version": self.experiment.evaluation_code_version,
            },
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required config key '{key}' in section '{ctx}'")
    return d[key]


def load_config_from_dict(
    raw: dict[str, Any],
    *,
    limit_override: int | None = None,
    batch_size_override: int | None = None,
) -> EvalConfig:
    """Parse a raw (YAML-loaded) dict into a validated EvalConfig."""
    if "model" not in raw:
        raise ValueError("Missing required config section 'model'")
    if "task" not in raw:
        raise ValueError("Missing required config section 'task'")

    m = raw.get("model", {})
    t = raw.get("task", {})
    p = raw.get("prompt", {})
    g = raw.get("generation", {})
    r = raw.get("runtime", {})
    e = raw.get("experiment", {})

    model_name = _require(m, "name", "model")
    intervention = m.get("intervention", {"type": "none"})
    if not isinstance(intervention, dict) or "type" not in intervention:
        raise ValueError("model.intervention must be a dict with a 'type' key")

    task_limit = t.get("limit", None)
    if limit_override is not None:
        task_limit = limit_override
    if task_limit is not None:
        task_limit = int(task_limit)
        if task_limit <= 0:
            raise ValueError("task.limit must be > 0 or null")

    batch_size = r.get("batch_size", 1)
    if batch_size_override is not None:
        batch_size = batch_size_override
    batch_size = int(batch_size)

    valid_splits = {"test", "train"}
    if t.get("split", "test") not in valid_splits:
        raise ValueError(f"task.split must be one of {valid_splits}")

    valid_devices = {"auto", "cpu", "mps", "cuda"}
    if m.get("device", "auto") not in valid_devices:
        raise ValueError(f"model.device must be one of {valid_devices}")

    backend = m.get("backend", "hf")
    if backend not in ("hf", "vllm"):
        raise ValueError("model.backend must be 'hf' or 'vllm'")
    tp = int(m.get("tensor_parallel_size", 1))
    if tp < 1:
        raise ValueError("model.tensor_parallel_size must be >= 1")
    gmu = float(m.get("gpu_memory_utilization", 0.9))
    if not (0.0 < gmu <= 1.0):
        raise ValueError("model.gpu_memory_utilization must be in (0, 1]")

    model = ModelConfig(
        name=model_name,
        backend=backend,
        revision=m.get("revision"),
        tokenizer_revision=m.get("tokenizer_revision"),
        dtype=m.get("dtype", "float32"),
        device=m.get("device", "auto"),
        trust_remote_code=bool(m.get("trust_remote_code", False)),
        use_chat_template=bool(m.get("use_chat_template", True)),
        attn_implementation=m.get("attn_implementation"),
        tensor_parallel_size=tp,
        gpu_memory_utilization=gmu,
        enforce_eager=bool(m.get("enforce_eager", False)),
        max_model_len=m.get("max_model_len"),
        intervention_type=str(intervention.get("type", "none")),
        intervention=dict(intervention),
    )
    task = TaskConfig(
        name=t.get("name", "gsm8k"),
        version=str(t.get("version", "1.0")),
        split=t.get("split", "test"),
        limit=task_limit,
        dataset_id=t.get("dataset_id", "openai/gsm8k"),
        dataset_config=t.get("dataset_config", "main"),
        dataset_revision=t.get("dataset_revision"),
    )
    prompt = PromptConfig(
        template_name=p.get("template_name", "gsm8k_cot_v1"),
        template_version=str(p.get("template_version", "1.0")),
    )
    generation = GenerationConfig(
        max_new_tokens=int(g.get("max_new_tokens", 512)),
        temperature=float(g.get("temperature", 0.0)),
        top_p=float(g.get("top_p", 1.0)),
        do_sample=bool(g.get("do_sample", False)),
        seed=int(g.get("seed", 42)),
    )
    runtime = RuntimeConfig(
        batch_size=batch_size,
        output_dir=str(r.get("output_dir", "results")),
        seed=int(r.get("seed", g.get("seed", 42))),
    )
    experiment = ExperimentConfig(
        name=str(e.get("name", "experiment")),
        evaluation_code_version=str(e.get("evaluation_code_version", "0.1.0")),
    )
    return EvalConfig(
        model=model, task=task, prompt=prompt,
        generation=generation, runtime=runtime, experiment=experiment,
    )


def load_config_from_yaml(
    path: str | Path,
    *,
    limit_override: int | None = None,
    batch_size_override: int | None = None,
) -> EvalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return load_config_from_dict(
        raw,
        limit_override=limit_override,
        batch_size_override=batch_size_override,
    )
