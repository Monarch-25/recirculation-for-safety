"""Unit tests: config parsing."""

import pytest

from eval_harness.core.config import InterventionConfig, load_config_from_dict


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


def test_intervention_none_default():
    cfg = load_config_from_dict(_base())
    assert cfg.model.intervention.type == "none"
    assert cfg.model.intervention_type == "none"
    assert cfg.model.intervention.to_dict() == {"type": "none"}


def test_intervention_none_with_extras_raises():
    raw = _base()
    raw["model"] = {"name": "m", "intervention": {
        "type": "none", "source_layer": 11}}
    with pytest.raises(ValueError, match="recirculation"):
        load_config_from_dict(raw)


def test_intervention_recirc_flat():
    cfg = load_config_from_dict({
        **_base(),
        "model": {"name": "m", "intervention": {
            "type": "recirculation", "source_layer": 18,
            "destination_layer": 9, "alpha": 0.15, "beta": 1.0,
            "normalization": "destination_l2"}},
    })
    iv = cfg.model.intervention
    assert (iv.source_layer, iv.destination_layer) == (18, 9)
    assert iv.effective_beta == 1.0


def test_intervention_nested_shapes_and_beta_modes():
    base = {"type": "recirculation", "source_layer": 11,
            "destination_layer": 4, "alpha": 0.15,
            "mixture": {"mode": "convex"},
            "normalization": {"type": "destination_l2"},
            "ramping": {"enabled": True, "tokens": 10}}
    convex = InterventionConfig.from_dict(base)
    assert convex.effective_beta == pytest.approx(0.85)
    assert convex.ramp_tokens == 10
    nonconvex = InterventionConfig.from_dict(
        {**base, "mixture": "nonconvex", "ramping": {"enabled": False}})
    assert nonconvex.effective_beta == 1.0
    assert nonconvex.ramp_tokens == 0


@pytest.mark.parametrize("bad", [
    {"type": "recirculation"},  # missing layers
    {"type": "recirculation", "source_layer": 4, "destination_layer": 4},
    {"type": "recirculation", "source_layer": 3, "destination_layer": 9},
    {"type": "recirculation", "source_layer": 11,
     "destination_layer": 4, "alpha": 1.5},
    {"type": "recirculation", "source_layer": 11,
     "destination_layer": 4, "mixture": "weird"},
    {"type": "recirculation", "source_layer": 11,
     "destination_layer": 4, "normalization": "weird"},
    {"type": "something-else"},
])
def test_bad_recirc_configs_raise(bad):
    with pytest.raises(ValueError, match="intervention"):
        InterventionConfig.from_dict(bad)
