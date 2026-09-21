"""vLLM-backed smoke test (Linux + GPU only). Run with: pytest -m vllm."""

import pytest

from eval_harness.core.interfaces import GenerationConfig

pytestmark = pytest.mark.vllm

vllm = pytest.importorskip("vllm", reason="vllm not installed (Linux-only)")


def test_vllm_adapter_generates():
    from eval_harness.models.vllm_adapter import VLLMModelAdapter
    adapter = VLLMModelAdapter(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        dtype="auto",
        enforce_eager=True,  # avoids CUDA-graph warmup issues on small GPUs
    )
    out = adapter.generate(
        ["Solve step by step. What is 2 + 2? Give the final answer clearly."],
        GenerationConfig(max_new_tokens=16, do_sample=False,
                         temperature=0.0, top_p=1.0, seed=42),
    )
    assert len(out) == 1 and isinstance(out[0], str) and len(out[0]) > 0
    assert adapter.get_metadata()["backend"] == "vllm"
