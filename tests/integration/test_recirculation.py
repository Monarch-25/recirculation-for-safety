"""Recirculation invariants on a real model (SmolLM2-360M, local).

Run with: pytest -m model. Weights resolve from the HF cache when present,
so this also runs on machines without large downloads after first fetch.
Covers plan P2 §23 (alpha=0 == baseline), §25, §33-34 (batch/determinism).
"""

import pytest

from eval_harness.core.interfaces import GenerationConfig

pytestmark = pytest.mark.model

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

PROMPTS = [
    "Q: What is 2 + 2? A: Let's think step by step.",
    "Q: A robe takes 2 bolts of blue fiber and half that much white "
    "fiber. How many bolts in total? A: Let's think step by step.",
]


def _gen(n=16):
    return GenerationConfig(max_new_tokens=n, do_sample=False,
                            temperature=0.0, top_p=1.0, seed=42)


def _common_kwargs():
    # Raw prompts (PT-style) + eager attention required on MPS/torch 2.6.
    return {"dtype": "float32", "device": "auto",
            "attn_implementation": "eager", "use_chat_template": False}


def _recirc_kwargs(adapter, alpha):
    n = len(adapter.model.model.layers)
    dest, src = 4, min(11, n - 1)
    assert src > dest, f"test model needs >5 blocks, got {n}"
    return {"intervention": {
        "type": "recirculation", "source_layer": src,
        "destination_layer": dest, "alpha": alpha,
        "mixture": {"mode": "convex"},
        "normalization": {"type": "destination_l2"},
        "ramping": {"enabled": False}}}


@pytest.fixture(scope="module")
def baseline():
    from eval_harness.models.hf_causal_lm import HFCausalLMAdapter
    return HFCausalLMAdapter(MODEL, **_common_kwargs())


@pytest.fixture(scope="module")
def recirc_alpha0(baseline):
    from eval_harness.models.recirculation import RecirculationModelAdapter
    return RecirculationModelAdapter(
        MODEL, **_common_kwargs(), **_recirc_kwargs(baseline, 0.0))


@pytest.fixture(scope="module")
def recirc_on(baseline):
    from eval_harness.models.recirculation import RecirculationModelAdapter
    return RecirculationModelAdapter(
        MODEL, **_common_kwargs(), **_recirc_kwargs(baseline, 0.15))


def test_alpha_zero_matches_baseline(recirc_alpha0, baseline):
    """P2 §23: recirculation(alpha=0) must reproduce baseline outputs."""
    got = recirc_alpha0.generate(PROMPTS, _gen())
    want = baseline.generate(PROMPTS, _gen())
    assert got == want


def test_determinism_twice_identical(recirc_on):
    first = recirc_on.generate(PROMPTS, _gen())
    second = recirc_on.generate(PROMPTS, _gen())
    assert first == second


def test_batch_size_invariance(recirc_on):
    together = recirc_on.generate(PROMPTS, _gen())
    separate = (recirc_on.generate(PROMPTS[:1], _gen())
                + recirc_on.generate(PROMPTS[1:], _gen()))
    assert together == separate


def test_nonzero_alpha_runs(recirc_on):
    out = recirc_on.generate(PROMPTS[:1], _gen())
    assert len(out) == 1 and len(out[0]) > 0
    meta = recirc_on.get_metadata()
    assert meta["layer_indexing_convention"] == "block_0based"
    assert meta["recurrence_variant"] == "cross_step_tokenwise_serial"
    assert meta["hidden_size"] and meta["num_layers"]


def test_debug_capture(recirc_on):
    recirc_on._debug_steps = 6
    try:
        recirc_on.generate(PROMPTS[:1], _gen())
        dbg = recirc_on.last_debug
    finally:
        recirc_on._debug_steps = 0
    assert 0 < len(dbg) <= 6
    for row in dbg:
        for key in ("position", "alpha_effective", "source_norm",
                    "destination_norm", "scaled_source_norm", "mixed_norm",
                    "cosine"):
            assert key in row, key
        assert -1.0 <= row["cosine"] <= 1.0
        assert row["source_norm"] > 0 and row["destination_norm"] > 0


def test_layer_bounds_rejected_without_loading():
    from eval_harness.models.recirculation import RecirculationModelAdapter
    with pytest.raises(ValueError, match="out of bounds"):
        RecirculationModelAdapter(
            MODEL, **_common_kwargs(),
            intervention={"type": "recirculation", "source_layer": 9999,
                          "destination_layer": 0, "alpha": 0.1})
