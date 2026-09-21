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


def test_two_stage_real_smoke(tmp_path):
    """End-to-end Kojima two-stage path on a real model (1 example)."""
    from eval_harness.core.config import load_config_from_dict
    from eval_harness.core.evaluator import Evaluator
    from eval_harness.models.hf_causal_lm import HFCausalLMAdapter
    from eval_harness.tasks.gsm8k import GSM8KTask
    cfg = load_config_from_dict({
        "model": {"name": "HuggingFaceTB/SmolLM2-360M-Instruct",
                  "dtype": "float32", "device": "auto",
                  "attn_implementation": "eager",
                  "use_chat_template": False,
                  "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test", "limit": 1},
        "prompt": {"template_name": "gsm8k_kojima_v1",
                   "extraction_template_name": "gsm8k_answer_extract_v1"},
        "generation": {"max_new_tokens": 32},
        "runtime": {"batch_size": 1, "output_dir": "results"},
        "experiment": {"name": "two-stage-smoke"},
    })
    task = GSM8KTask(template_name="gsm8k_kojima_v1",
                     extraction_template_name="gsm8k_answer_extract_v1")
    model = HFCausalLMAdapter(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        dtype="float32", device="auto", attn_implementation="eager",
        use_chat_template=False)
    res = Evaluator(output_root=tmp_path / "results").evaluate(
        task=task, model=model, config=cfg,
        run_id="twostage", command="pytest")
    assert res.metrics.num_examples == 1
    rec = res.records[0]
    assert rec.reasoning and len(rec.reasoning) > 0
    assert rec.raw_output and len(rec.raw_output) > 0
    assert "arabic numerals" in (rec.extraction_prompt or "")
    ext = res.manifest["prompt"]["extraction"]
    assert ext["name"] == "gsm8k_answer_extract_v1" and ext["hash"]
