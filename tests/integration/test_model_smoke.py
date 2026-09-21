"""Model-backed smoke test (requires download). Run with: pytest -m model."""

import pytest

from eval_harness.core.config import GenerationConfig

pytestmark = pytest.mark.model


def test_hf_adapter_generates():
    from eval_harness.models.hf_causal_lm import HFCausalLMAdapter
    adapter = HFCausalLMAdapter(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        dtype="float32",
        device="auto",
        # Eager attention required on MPS with torch 2.6 (see protocol doc).
        attn_implementation="eager",
    )
    assert adapter.num_parameters < 500_000_000
    out = adapter.generate(
        ["Solve step by step. What is 2 + 2? Give the final answer clearly."],
        GenerationConfig(max_new_tokens=32, do_sample=False,
                         temperature=0.0, top_p=1.0, seed=42),
    )
    assert len(out) == 1 and isinstance(out[0], str) and len(out[0]) > 0
