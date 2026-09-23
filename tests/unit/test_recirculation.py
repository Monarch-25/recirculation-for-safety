"""Unit tests: recirculation mixing math + helpers (fully offline)."""

import pytest
import torch

from eval_harness.core.config import (
    InterventionConfig,
    normalize_intervention,
)
from eval_harness.models.recirculation import (
    _resolve_decoder_layers,
    mix_destination,
    mix_destination_batched,
    ramp_batch,
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
    # Paper App. B.3: factor = min(t/10, 1); t=0 mixes nothing.
    assert ramp_factor(0, 10) == pytest.approx(0.0)
    assert ramp_factor(1, 10) == pytest.approx(0.1)
    assert ramp_factor(9, 10) == pytest.approx(0.9)
    assert ramp_factor(10, 10) == pytest.approx(1.0)
    assert ramp_factor(500, 10) == pytest.approx(1.0)
    assert ramp_factor(3, 0) == pytest.approx(1.0)
    assert ramp_factor(3, -1) == pytest.approx(1.0)


def test_ramp_batch_matches_scalar():
    pos = torch.tensor([0, 4, 9, 99])
    got = ramp_batch(pos, 10)
    want = torch.tensor([ramp_factor(int(p), 10) for p in pos])
    assert torch.allclose(got, want)
    assert torch.all(ramp_batch(pos, 0) == 1.0)


def test_batched_mix_matches_single_row():
    torch.manual_seed(0)
    dst = torch.randn(3, 8)
    src = torch.randn(3, 8)
    alpha = torch.tensor([0.0, 0.5, 0.15])
    got = mix_destination_batched(dst, src, alpha, 0.85)
    for r in range(3):
        want = mix_destination(dst[r], src[r], float(alpha[r]), 0.85)
        assert torch.allclose(got[r], want), r
    # Scalar alpha broadcasts.
    got2 = mix_destination_batched(dst, src, 0.15, 1.0,
                                   normalization="identity")
    assert torch.allclose(got2, 0.15 * src + dst)


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


def test_mozer_schedule_round_trip():
    cfg = InterventionConfig.from_dict({
        "type": "recirculation", "source_layer": 25,
        "destination_layer": 20, "alpha": 0.04,
        "schedule": "mozer",
        "mixture": {"mode": "convex"},
        "normalization": {"type": "destination_l2"},
        "ramping": {"enabled": False},
    })
    assert cfg.schedule == "mozer"
    assert cfg.effective_beta == pytest.approx(0.96)
    assert InterventionConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.to_dict()["schedule"] == "mozer"


def test_mozer_schedule_invalid_raises():
    from eval_harness.core.config import InterventionConfig
    with pytest.raises(ValueError, match="schedule"):
        InterventionConfig.from_dict({
            "type": "recirculation", "source_layer": 25,
            "destination_layer": 20, "schedule": "zap"})


def test_batched_beta_tensor_coupling():
    """Paper convex coupling beta_t = 1 - alpha_t realized per-row."""
    torch.manual_seed(0)
    dst = torch.randn(4, 8)
    src = torch.randn(4, 8)
    alpha = torch.tensor([0.0, 0.04, 0.1, 0.15])
    beta = 1.0 - alpha  # convex-coupled beta vector
    got = mix_destination_batched(dst, src, alpha, beta)
    for r in range(4):
        want = mix_destination(dst[r], src[r], float(alpha[r]),
                               float(beta[r]))
        assert torch.allclose(got[r], want), r


def test_batched_beta_tensor_passthrough():
    """Beta vector vs scalar must agree on identical values."""
    torch.manual_seed(0)
    dst = torch.randn(3, 8)
    src = torch.randn(3, 8)
    alpha = torch.tensor([0.04, 0.04, 0.04])
    got_v = mix_destination_batched(dst, src, alpha,
                                    torch.tensor([1.0, 1.0, 1.0]))
    got_s = mix_destination_batched(dst, src, alpha, 1.0)
    assert torch.allclose(got_v, got_s)


def test_normalize_intervention_shapes():
    assert normalize_intervention(None) == {"type": "none"}
    assert normalize_intervention({"type": "none"}) == {"type": "none"}
    typed = InterventionConfig.from_dict(
        {"type": "recirculation", "source_layer": 11,
         "destination_layer": 4})
    assert normalize_intervention(typed)["source_layer"] == 11
    with pytest.raises(TypeError):
        normalize_intervention("recirculation")


class _FakeCache:
    def __init__(self, layers=3, batch=2, kv=4, dim=8):
        import torch
        self.key_cache = [torch.randn(batch, kv, 5, dim)
                          for _ in range(layers)]
        self.value_cache = [torch.randn(batch, kv, 5, dim)
                            for _ in range(layers)]


def test_truncate_cache_keeps_prefix():
    from eval_harness.models.recirculation import RecirculationModelAdapter
    cache = _FakeCache()
    RecirculationModelAdapter._truncate_cache(cache, 3)
    for stack in (cache.key_cache, cache.value_cache):
        for t in stack:
            assert t.shape[2] == 3


def test_truncate_cache_prefers_native_crop():
    from eval_harness.models.recirculation import RecirculationModelAdapter

    class NativeCache:
        def __init__(self):
            self.cropped_to = None

        def crop(self, length):
            self.cropped_to = length

    cache = NativeCache()
    RecirculationModelAdapter._truncate_cache(cache, 4)
    # Redo removes exactly the last-added token (remove-last), not the
    # absolute length: Gemma3 sliding layers cap stored length below
    # `length` once the window is full, where absolute crop is a no-op
    # or raises.
    assert cache.cropped_to == -1


def test_truncate_cache_rejects_foreign_caches():
    from eval_harness.models.recirculation import RecirculationModelAdapter
    with pytest.raises(TypeError, match="croppable KV cache"):
        RecirculationModelAdapter._truncate_cache(object(), 2)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_resolve_llama_like_layout():
    model = _NS(model=_NS(layers=[1, 2, 3]))
    assert _resolve_decoder_layers(model, "m") == [1, 2, 3]


def test_resolve_gemma3_wrapper_layout():
    # Gemma3ForCausalLM.model is a Gemma3Model multimodal wrapper; the
    # text stack lives under .language_model (this exact shape crashed
    # the first 4B run: 'Gemma3Model' has no attribute 'layers').
    text = _NS(layers=[1] * 34, embed_tokens=_NS(weight=[[0]]))
    model = _NS(model=_NS(language_model=text))
    assert _resolve_decoder_layers(model, "m") == [1] * 34


def test_resolve_gpt2_like_layout():
    model = _NS(transformer=_NS(h=[1, 2]))
    assert _resolve_decoder_layers(model, "m") == [1, 2]


def test_resolve_unknown_layout_raises():
    with pytest.raises(RuntimeError, match="known decoder-block layout"):
        _resolve_decoder_layers(_NS(), "m")


def test_apply_stop_strings_truncates_earliest():
    from eval_harness.models.recirculation import apply_stop_strings
    assert apply_stop_strings("answer 6\nQ: next", ["Q:", "</s>"]) == "answer 6\n"
    assert apply_stop_strings("clean text", ["Q:"]) == "clean text"
    assert apply_stop_strings("a</s>b", []) == "a</s>b"


def test_eos_token_ids_prefers_model_config_list():
    from eval_harness.models.hf_causal_lm import HFCausalLMAdapter

    class Cfg:
        eos_token_id = [1, 106]

    class Model:
        config = Cfg()

    class Tok:
        eos_token_id = 1

    adapter = HFCausalLMAdapter.__new__(HFCausalLMAdapter)
    adapter.model = Model()
    adapter.tokenizer = Tok()
    assert adapter.eos_token_ids() == [1, 106]

    class Cfg2:
        eos_token_id = None

    adapter.model = type("M", (), {"config": Cfg2()})()
    assert adapter.eos_token_ids() == [1]
