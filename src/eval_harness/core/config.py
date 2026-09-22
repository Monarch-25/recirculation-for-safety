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
class InterventionConfig:
    """Typed inference-intervention configuration.

    ``type="none"`` is the baseline (first-class experimental condition).
    ``type="recirculation"`` is the fixed training-free deep-to-shallow
    intervention (paper v2 default: convex mixture, destination-L2
    rescaling, 0-based transformer-block indexing with source > dest).

    Accepts both flat and nested YAML shapes, e.g.::

        intervention:
          type: recirculation
          source_layer: 18
          destination_layer: 9
          alpha: 0.15
          mixture: {mode: nonconvex}        # or: mixture: nonconvex
          normalization: {type: destination_l2}  # or a plain string
          ramping: {enabled: false, tokens: 10}  # or: ramp_tokens: 10
    """

    type: str = "none"
    source_layer: int | None = None
    destination_layer: int | None = None
    alpha: float = 0.15
    beta: float | None = None
    mixture: str = "convex"  # convex | nonconvex
    normalization: str = "destination_l2"  # destination_l2 | identity
    ramp_tokens: int = 0
    # Recurrence schedule (recirculation only):
    # - cross_step: one pass per input step; destination at t+1 mixes the
    #   stored deep source from input step t (plan §26 reading).
    # - two_pass: per input step, a normal full pass captures the source,
    #   then the upper stack reruns from the mixed boundary (same-step
    #   source, KV overwritten) and supplies the logits. ~2x compute.
    schedule: str = "cross_step"  # cross_step | two_pass

    @property
    def effective_beta(self) -> float:
        """Resolved destination coefficient.

        Explicit ``beta`` wins; otherwise convex → ``1 - alpha``,
        nonconvex → ``1.0`` (paper: 4B/12B critically require beta=1).
        """
        if self.beta is not None:
            return self.beta
        if self.mixture == "convex":
            return 1.0 - self.alpha
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        if self.type == "none":
            return {"type": "none"}
        return {
            "type": self.type,
            "source_layer": self.source_layer,
            "destination_layer": self.destination_layer,
            "alpha": self.alpha,
            "beta": self.beta,
            "effective_beta": self.effective_beta,
            "mixture": {"mode": self.mixture},
            "normalization": {"type": self.normalization},
            "ramping": {
                "enabled": self.ramp_tokens > 0,
                "tokens": self.ramp_tokens,
            },
            "schedule": self.schedule,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "InterventionConfig":
        raw = raw if raw is not None else {"type": "none"}
        if not isinstance(raw, dict) or "type" not in raw:
            raise ValueError(
                "model.intervention must be a dict with a 'type' key")
        itype = str(raw.get("type", "none"))
        if itype not in ("none", "recirculation"):
            raise ValueError(
                "model.intervention.type must be 'none' or "
                f"'recirculation', got {itype!r}")
        if itype == "none":
            extra = set(raw) - {"type"}
            if extra:
                raise ValueError(
                    f"model.intervention keys {sorted(extra)} require "
                    "type='recirculation', got type='none'")
            return cls()

        # -- recirculation: required layers --------------------------------
        try:
            src = int(raw["source_layer"])
            dst = int(raw["destination_layer"])
        except KeyError as exc:
            raise ValueError(
                f"model.intervention missing required key {exc} "
                "for type='recirculation'") from exc
        if src < 0 or dst < 0:
            raise ValueError(
                "model.intervention layers must be >= 0 "
                f"(got source={src}, dest={dst})")
        if src <= dst:
            raise ValueError(
                "model.intervention requires source_layer > "
                f"destination_layer (got source={src}, dest={dst}); "
                "source==destination is ambiguous and rejected")

        alpha = float(raw.get("alpha", 0.15))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"model.intervention.alpha must be in [0, 1] (got {alpha})")
        beta = raw.get("beta", None)
        beta = None if beta is None else float(beta)
        if beta is not None and beta < 0.0:
            raise ValueError(
                f"model.intervention.beta must be >= 0 (got {beta})")

        mixture = raw.get("mixture", "convex")
        if isinstance(mixture, dict):
            mixture = mixture.get("mode", "convex")
        mixture = str(mixture)
        if mixture not in ("convex", "nonconvex"):
            raise ValueError(
                "model.intervention.mixture mode must be 'convex' or "
                f"'nonconvex' (got {mixture!r})")

        norm = raw.get("normalization", "destination_l2")
        if isinstance(norm, dict):
            norm = norm.get("type", "destination_l2")
        norm = str(norm)
        if norm not in ("destination_l2", "identity"):
            raise ValueError(
                "model.intervention.normalization type must be "
                f"'destination_l2' or 'identity' (got {norm!r})")

        ramp_tokens = 0
        if "ramp_tokens" in raw:
            ramp_tokens = int(raw["ramp_tokens"])
        else:
            ramp = raw.get("ramping", {}) or {}
            if not isinstance(ramp, dict):
                raise ValueError(
                    "model.intervention.ramping must be a dict with "
                    "'enabled'/'tokens' keys")
            if bool(ramp.get("enabled", False)):
                ramp_tokens = int(ramp.get("tokens", 10))
        if ramp_tokens < 0:
            raise ValueError(
                "model.intervention ramp tokens must be >= 0 "
                f"(got {ramp_tokens})")

        schedule = str(raw.get("schedule", "cross_step"))
        if schedule not in ("cross_step", "two_pass"):
            raise ValueError(
                "model.intervention.schedule must be 'cross_step' or "
                f"'two_pass' (got {schedule!r})")

        return cls(
            type="recirculation",
            source_layer=src,
            destination_layer=dst,
            alpha=alpha,
            beta=beta,
            mixture=mixture,
            normalization=norm,
            ramp_tokens=ramp_tokens,
            schedule=schedule,
        )


def normalize_intervention(
    value: "InterventionConfig | dict[str, Any] | None",
) -> dict[str, Any]:
    """Coerce an intervention spec to a plain metadata dict.

    Model adapters accept the typed config, a raw dict, or None
    (baseline) and record a plain dict for manifest metadata.
    """
    if value is None:
        return {"type": "none"}
    if isinstance(value, InterventionConfig):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        "intervention must be InterventionConfig, dict, or None, "
        f"got {type(value).__name__}")


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
    intervention: InterventionConfig = field(
        default_factory=InterventionConfig)

    @property
    def intervention_type(self) -> str:
        """Baseline reports 'none'; recirculation reports 'recirculation'."""
        return self.intervention.type


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
    # Second-stage answer-extraction template (Kojima "[X'] [Z] [A]").
    # None = single-stage evaluation.
    extraction_template_name: str | None = None
    extraction_template_version: str = "1.0"


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
    # W&B tracking (P2 research logging). None = disabled.
    wandb_project: str | None = None
    wandb_entity: str | None = None


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
                "intervention": self.model.intervention.to_dict(),
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
                "extraction_template_name":
                    self.prompt.extraction_template_name,
                "extraction_template_version":
                    self.prompt.extraction_template_version,
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
                "wandb_project": self.experiment.wandb_project,
                "wandb_entity": self.experiment.wandb_entity,
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
    intervention = InterventionConfig.from_dict(
        m.get("intervention", {"type": "none"}))

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
        intervention=intervention,
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
        extraction_template_name=p.get("extraction_template_name"),
        extraction_template_version=str(
            p.get("extraction_template_version", "1.0")),
    )
    generation = GenerationConfig(
        max_new_tokens=int(g.get("max_new_tokens", 512)),
        temperature=float(g.get("temperature", 0.0)),
        top_p=float(g.get("top_p", 1.0)),
        do_sample=bool(g.get("do_sample", False)),
        seed=int(g.get("seed", 42)),
        extraction_max_new_tokens=(
            None if g.get("extraction_max_new_tokens") is None
            else int(g.get("extraction_max_new_tokens"))),
    )
    runtime = RuntimeConfig(
        batch_size=batch_size,
        output_dir=str(r.get("output_dir", "results")),
        seed=int(r.get("seed", g.get("seed", 42))),
    )
    experiment = ExperimentConfig(
        name=str(e.get("name", "experiment")),
        evaluation_code_version=str(e.get("evaluation_code_version", "0.1.0")),
        wandb_project=e.get("wandb_project"),
        wandb_entity=e.get("wandb_entity"),
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
