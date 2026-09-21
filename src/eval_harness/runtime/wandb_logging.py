"""Weights & Biases experiment tracking (optional, lazy, config-gated).

Enabled per-run via ``experiment.wandb_project`` (null = disabled).
``wandb`` is an optional dependency (``pip install -e ".[tracking]"``);
every ``wandb`` import lives inside functions so the offline suite and
CUDA-free paths never touch it. Auth comes from ``WANDB_API_KEY`` in the
environment (Modal ``wandb`` secret or a local ``wandb login``) — never
from files or configs.

Logging must never crash an evaluation: artifacts on disk are primary,
so every entry point catches its own errors, warns, and returns None.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def is_enabled(experiment: Any) -> bool:
    return bool(getattr(experiment, "wandb_project", None))


def _has_auth() -> bool:
    """True when a key is reachable without prompting.

    Avoids interactive `wandb login` prompts inside batch jobs: explicit
    env key wins, else a pre-existing local login counts.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    for candidate in ("~/.netrc", "~/.config/wandb/settings"):
        try:
            if Path(candidate).expanduser().exists():
                return True
        except Exception:
            pass
    return False


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Small JSON-safe slice of the eval config for wandb.config."""
    try:
        model = config.model
        intervention = model.intervention
        iv = intervention.to_dict() if hasattr(intervention, "to_dict") \
            else dict(intervention)
    except Exception:
        iv = None
    try:
        return {
            "model": getattr(config.model, "name", None),
            "model_revision": getattr(config.model, "revision", None),
            "backend": getattr(config.model, "backend", None),
            "dtype": getattr(config.model, "dtype", None),
            "intervention": iv,
            "task": getattr(config.task, "name", None),
            "task_version": getattr(config.task, "version", None),
            "split": getattr(config.task, "split", None),
            "limit": getattr(config.task, "limit", None),
            "dataset_revision": getattr(config.task, "dataset_revision", None),
            "prompt": getattr(config.prompt, "template_name", None),
            "prompt_version": getattr(config.prompt, "template_version", None),
            "generation": config.generation.to_dict()
            if hasattr(config.generation, "to_dict") else None,
            "batch_size": getattr(config.runtime, "batch_size", None),
            "seed": getattr(config.generation, "seed", None),
            "experiment": getattr(config.experiment, "name", None),
        }
    except Exception as exc:
        log.warning("wandb config snapshot failed: %s", exc)
        return {}


def log_evaluation(
    config: Any,
    *,
    run_id: str,
    run_dir: str,
    metrics: Any,
    performance: dict[str, Any] | None = None,
) -> str | None:
    """Log one completed evaluation. Returns the run URL or None."""
    exp = getattr(config, "experiment", None)
    if not is_enabled(exp):
        return None
    if not _has_auth():
        log.warning("wandb_project set but no WANDB_API_KEY and no saved "
                    "login; skipping (wandb login or set the key)")
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb_project set but wandb is not installed "
                    "(pip install -e \".[tracking]\"); skipping")
        return None
    try:
        metrics_d = metrics.to_dict() if hasattr(metrics, "to_dict") \
            else dict(metrics)
        perf = {f"perf/{k}": v for k, v in (performance or {}).items()
                if v is not None}
        token_cov = (performance or {}).get("token_stats_complete")
        run = wandb.init(
            project=exp.wandb_project,
            entity=getattr(exp, "wandb_entity", None),
            name=f"{getattr(exp, 'name', 'eval')}-{run_id}",
            job_type="evaluation",
            dir=run_dir,
            config=_config_snapshot(config),
            tags=[f"task:{getattr(config.task, 'name', '?')}",
                  f"backend:{getattr(config.model, 'backend', '?')}",
                  f"iv:{getattr(config.model.intervention, 'type', '?')}"],
        )
        wandb.log({**metrics_d, **perf,
                   "token_stats_complete": float(bool(token_cov))})
        url = getattr(run, "url", None)
        wandb.finish()
        log.info("wandb logged: %s", url)
        return url
    except Exception as exc:
        log.warning("wandb logging failed (eval artifacts unaffected): %s",
                    exc)
        try:
            import wandb
            wandb.finish(exit_code=1)
        except Exception:
            pass
        return None


def log_comparison(
    project: str,
    summary: dict[str, Any],
    *,
    entity: str | None = None,
    name: str | None = None,
) -> str | None:
    """Log a paired-comparison summary as its own run. Returns URL/None."""
    if not project:
        return None
    if not _has_auth():
        log.warning("wandb comparison requested but no auth; skipping")
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb requested but not installed; skipping")
        return None
    try:
        p = summary.get("paired", {})
        cells = p.get("cells", {})
        flat: dict[str, Any] = {
            "baseline_accuracy": p.get("baseline_accuracy"),
            "treatment_accuracy": p.get("treatment_accuracy"),
            "absolute_delta": p.get("absolute_delta"),
            "relative_delta": p.get("relative_delta"),
            "rescued": p.get("rescued_examples"),
            "regressed": p.get("regressed_examples"),
            "n_common": p.get("n_common"),
            "n_changed": summary.get("n_changed"),
            "cc": cells.get("cc"), "cw": cells.get("cw"),
            "wc": cells.get("wc"), "ww": cells.get("ww"),
        }
        for cond in ("baseline", "treatment"):
            ln = (summary.get("lengths", {}) or {}).get(cond, {})
            for stat in ("chars", "words"):
                for agg in ("mean", "median", "p90"):
                    flat[f"len_{cond}_{stat}_{agg}"] = (
                        ln.get(stat, {}) or {}).get(agg)
            flat[f"len_{cond}_parse_failures"] = ln.get("parse_failures")
        flat = {k: v for k, v in flat.items() if v is not None}
        run = wandb.init(
            project=project, entity=entity,
            name=name or "compare",
            job_type="comparison",
            config={
                "baseline_run": summary.get("baseline", {}).get("run_dir"),
                "treatment_run": summary.get("treatment", {}).get("run_dir"),
                "baseline_model": summary.get("baseline", {}).get("model"),
                "treatment_model": summary.get("treatment", {}).get("model"),
                "treatment_intervention": summary.get(
                    "treatment", {}).get("intervention"),
            },
            tags=["comparison"],
        )
        wandb.log(flat)
        url = getattr(run, "url", None)
        wandb.finish()
        log.info("wandb comparison logged: %s", url)
        return url
    except Exception as exc:
        log.warning("wandb comparison logging failed: %s", exc)
        try:
            import wandb
            wandb.finish(exit_code=1)
        except Exception:
            pass
        return None
