"""Unit tests: recirculation mixing math + helpers (fully offline)."""

import pytest
import torch

from eval_harness.core.config import (
    InterventionConfig,
    normalize_intervention,
)
from eval_harness.models.recirculation import (
    mix_destination,
    ramp_factor,
    rescale_to_destination,
)


def test_rescale_matches_destination_norm():
    src = torch.tensor([0.0, 2.0])
    dst = torch.tensor([1.0, 0.0])
    out = rescale_to_destination(src, dst)
    assert torch.allclose(out, torch.tensor([0.0, 1.0]))
    assert out.dtype == src.dtype


def test_rescale_zero_source_safe():
    out = rescale_to_destination(
        torch.zeros(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.all(torch.isfinite(out))
    assert torch.all(out == 0)


def test_mix_convex_by_hand():
    dst = torch.tensor([1.0, 0.0])
    src = torch.tensor([0.0, 2.0])  # rescales to [0, 1]
    out = mix_destination(dst, src, alpha=0.5, beta=0.5)
    assert torch.allclose(out, torch.tensor([0.5, 0.5]))


def test_mix_identity_skips_rescale():
    dst = torch.tensor([1.0, 1.0])
    src = torch.tensor([10.0, 0.0])
    out = mix_destination(dst, src, alpha=1.0, beta=0.0,
                          normalization="identity")
    assert torch.allclose(out, src)


def test_mix_unknown_norm_raises():
    with pytest.raises(ValueError, match="normalization"):
        mix_destination(torch.ones(2), torch.ones(2), 0.1, 0.9,
                        normalization="weird")


def test_mix_alpha_zero_is_identity():
    dst = torch.randn(8)
    out = mix_destination(dst, torch.randn(8), alpha=0.0, beta=1.0)
    assert torch.allclose(out, dst)


def test_ramp_factor():
    assert ramp_factor(0, 10) == pytest.approx(0.1)
    assert ramp_factor(9, 10) == pytest.approx(1.0)
    assert ramp_factor(500, 10) == pytest.approx(1.0)
    assert ramp_factor(3, 0) == pytest.approx(1.0)
    assert ramp_factor(3, -1) == pytest.approx(1.0)


def test_intervention_round_trip():
    cfg = InterventionConfig.from_dict({
        "type": "recirculation", "source_layer": 18,
        "destination_layer": 9, "alpha": 0.15,
        "mixture": {"mode": "nonconvex"},
        "normalization": {"type": "destination_l2"},
        "ramping": {"enabled": True, "tokens": 10},
    })
    assert InterventionConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.to_dict()["effective_beta"] == 1.0


def test_normalize_intervention_shapes():
    assert normalize_intervention(None) == {"type": "none"}
    assert normalize_intervention({"type": "none"}) == {"type": "none"}
    typed = InterventionConfig.from_dict(
        {"type": "recirculation", "source_layer": 11,
         "destination_layer": 4})
    assert normalize_intervention(typed)["source_layer"] == 11
    with pytest.raises(TypeError):
        normalize_intervention("recirculation")
