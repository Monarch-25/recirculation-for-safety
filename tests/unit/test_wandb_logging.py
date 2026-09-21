"""Unit tests: wandb logging (offline-safe, never hits the network)."""

from eval_harness.core.config import load_config_from_dict
from eval_harness.core.results import AggregatedMetrics
from eval_harness.runtime import wandb_logging


def _cfg(project=None):
    return load_config_from_dict({
        "model": {"name": "mock/m", "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test"},
        "prompt": {}, "generation": {}, "runtime": {},
        "experiment": {"name": "t", **(
            {"wandb_project": project} if project else {})},
    })


def _metrics():
    return AggregatedMetrics(num_examples=2, num_correct=1, accuracy=0.5,
                             parse_rate=1.0, num_parse_failures=0)


def test_disabled_by_default(tmp_path):
    assert wandb_logging.is_enabled(_cfg().experiment) is False
    out = wandb_logging.log_evaluation(
        _cfg(), run_id="r", run_dir=str(tmp_path),
        metrics=_metrics(), performance={})
    assert out is None


def test_enabled_without_auth_skips(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc here
    cfg = _cfg(project="some-project")
    assert wandb_logging.is_enabled(cfg.experiment) is True
    out = wandb_logging.log_evaluation(
        cfg, run_id="r", run_dir=str(tmp_path),
        metrics=_metrics(), performance={"wall_clock_seconds": 1.0})
    assert out is None


def test_offline_mode_exercises_init_path(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(wandb_logging, "_has_auth", lambda: True)
    cfg = _cfg(project="offline-probe")
    out = wandb_logging.log_evaluation(
        cfg, run_id="r", run_dir=str(tmp_path),
        metrics=_metrics(),
        performance={"wall_clock_seconds": 1.0,
                     "token_stats_complete": True})
    assert out is None or isinstance(out, str)
    assert (tmp_path / "wandb").exists()


def test_comparison_guards(monkeypatch):
    assert wandb_logging.log_comparison("", {}) is None
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_logging.log_comparison(
        "proj", {"paired": {}}) is None
