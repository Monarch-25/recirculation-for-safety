"""Unit tests: GSM8K extraction-prompt composition (offline)."""

from eval_harness.core.interfaces import EvalExample
from eval_harness.tasks.gsm8k import GSM8KTask


def _example():
    return EvalExample(example_id="e", index=0,
                       question="What is 2 + 2?",
                       reference_answer="#### 4")


def test_no_extraction_template_is_single_stage():
    task = GSM8KTask()
    assert task.build_extraction_prompt(_example(), "some reasoning") is None


def test_extraction_composes_kojima_stages():
    task = GSM8KTask(
        template_name="gsm8k_kojima_v1",
        extraction_template_name="gsm8k_answer_extract_v1")
    out = task.build_extraction_prompt(_example(), "2 + 2 is 4.")
    assert out is not None
    assert "Q: What is 2 + 2?" in out
    assert "2 + 2 is 4." in out
    assert out.endswith("Therefore, the answer (arabic numerals) is")


def test_from_examples_passes_extraction_through():
    task, examples = GSM8KTask.from_examples(
        [{"question": "q?", "answer": "#### 1"}],
        extraction_template_name="gsm8k_answer_extract_v1")
    assert task.build_extraction_prompt(examples[0], "z") is not None
