"""Unit tests: config parsing."""

import pytest

from eval_harness.core.config import load_config_from_dict


def _base():
    return {
        "model": {"name": "org/model"},
        "task": {"name": "gsm8k", "split": "test"},
        "prompt": {},
        "generation": {},
        "runtime": {},
        "experiment": {},
    }


def test_minimal_config_defaults():
    cfg = load_config_from_dict(_base())
    assert cfg.model.name == "org/model"
    assert cfg.task.split == "test"
    assert cfg.generation.do_sample is False
    assert cfg.runtime.batch_size == 1


def test_missing_model_raises():
    raw = _base()
    del raw["model"]
    with pytest.raises(ValueError, match="model"):
        load_config_from_dict(raw)


def test_invalid_batch_size_raises():
    raw = _base()
    raw["runtime"] = {"batch_size": 0}
    with pytest.raises(ValueError, match="batch_size"):
        load_config_from_dict(raw)


def test_invalid_split_raises():
    raw = _base()
    raw["task"] = {"name": "gsm8k", "split": "dev"}
    with pytest.raises(ValueError, match="split"):
        load_config_from_dict(raw)


def test_invalid_device_raises():
    raw = _base()
    raw["model"] = {"name": "org/model", "device": "tpu"}
    with pytest.raises(ValueError, match="device"):
        load_config_from_dict(raw)


def test_overrides():
    cfg = load_config_from_dict(_base(), limit_override=5, batch_size_override=2)
    assert cfg.task.limit == 5
    assert cfg.runtime.batch_size == 2


def test_bad_intervention_raises():
    raw = _base()
    raw["model"] = {"name": "org/model", "intervention": {"no_type": 1}}
    with pytest.raises(ValueError, match="intervention"):
        load_config_from_dict(raw)


def test_generation_validation():
    raw = _base()
    raw["generation"] = {"max_new_tokens": 0}
    with pytest.raises(ValueError, match="max_new_tokens"):
        load_config_from_dict(raw)


def test_attn_implementation_passthrough():
    raw = _base()
    assert load_config_from_dict(raw).model.attn_implementation is None
    raw["model"] = {"name": "org/model", "attn_implementation": "eager"}
    assert load_config_from_dict(raw).model.attn_implementation == "eager"
