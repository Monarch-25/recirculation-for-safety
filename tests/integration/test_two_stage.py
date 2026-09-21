"""Integration: two-stage evaluation through the evaluator (offline).

Stage 1 (reasoning) and stage 2 (extraction) both flow through the same
mock adapter in order; the evaluator must keep example alignment across
both stages and all batch sizes.
"""

import json
from pathlib import Path

from eval_harness.core.config import load_config_from_dict
from eval_harness.core.evaluator import Evaluator
from eval_harness.models.mock import MockModelAdapter
from tests.integration.test_evaluator_mock import ListTask

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gsm8k_tiny.json"


class ExtractTask(ListTask):
    """ListTask + Kojima-style extraction stage."""

    def build_extraction_prompt(self, example, reasoning):
        return (f"{self.build_prompt(example)} || {reasoning} "
                f"|| Therefore, the answer (arabic numerals) is")


class FlakyExtractTask(ListTask):
    def build_extraction_prompt(self, example, reasoning):
        return None if example.index == 0 else "prompt"


def _rows(n=3):
    return json.loads(FIXTURE.read_text())[:n]


def _config(batch_size=2):
    return load_config_from_dict({
        "model": {"name": "mock/m", "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test"},
        "prompt": {"extraction_template_name": "gsm8k_answer_extract_v1"},
        "generation": {"max_new_tokens": 32,
                       "extraction_max_new_tokens": 7},
        "runtime": {"batch_size": batch_size, "output_dir": "results"},
        "experiment": {"name": "two-stage-test"},
    })


def test_two_stage_records_reasoning_and_final(tmp_path):
    rows = _rows()
    # Stage 1 reasoning, then stage-2 finals (first wrong, rest right).
    model = MockModelAdapter(["r1", "r2", "r3",
                              "blah no numbers", "#### 10", "#### 5"])
    ev = Evaluator(output_root=tmp_path / "results")
    res = ev.evaluate(task=ExtractTask(rows), model=model,
                      config=_config(), run_id="ts", command="pytest")
    assert [r.reasoning for r in res.records] == ["r1", "r2", "r3"]
    assert [r.raw_output for r in res.records] == [
        "blah no numbers", "#### 10", "#### 5"]
    assert res.records[0].correct is False
    assert res.records[1].correct is True
    assert all(r.extraction_prompt and "arabic numerals" in r.extraction_prompt
               for r in res.records)
    ext = res.manifest["prompt"]["extraction"]
    assert ext["name"] == "gsm8k_answer_extract_v1"
    assert ext["hash"]
    # Token totals span both stages (mock reports none -> nulls, honest).
    assert res.manifest["performance"]["token_stats_complete"] is False


def test_stage_token_budgets_reach_adapter(tmp_path):
    rows = _rows()[:2]
    model = MockModelAdapter(["r1", "r2", "a1", "a2"])
    Evaluator(output_root=tmp_path / "results").evaluate(
        task=ExtractTask(rows), model=model,
        config=_config(), run_id="caps", command="pytest")
    caps = [c["max_new_tokens"] for c in model.calls]
    assert caps == [32, 7]  # stage 1 reasoning, then stage 2 extraction


def test_two_stage_batch_invariance(tmp_path):
    rows = _rows()
    outs = ["r1", "r2", "r3", "#### 72", "#### 10", "#### 5"]
    maps = {}
    for bs in (1, 3):
        res = Evaluator(output_root=tmp_path / f"r{bs}").evaluate(
            task=ExtractTask(rows), model=MockModelAdapter(list(outs)),
            config=_config(bs), run_id=f"bs{bs}", command="pytest")
        maps[bs] = {r.example_id: (r.reasoning, r.raw_output, r.correct)
                    for r in res.records}
    assert maps[1] == maps[3]


def test_single_stage_unchanged_without_method(tmp_path):
    rows = _rows()[:2]
    res = Evaluator(output_root=tmp_path / "results").evaluate(
        task=ListTask(rows), model=MockModelAdapter(["#### 72", "#### 10"]),
        config=_config(), run_id="ss", command="pytest")
    assert all(r.reasoning is None and r.extraction_prompt is None
               for r in res.records)
    assert res.manifest["prompt"]["extraction"] is None


def test_partial_extraction_none_raises(tmp_path):
    rows = _rows()[:2]
    ev = Evaluator(output_root=tmp_path / "results")
    try:
        ev.evaluate(task=FlakyExtractTask(rows),
                    model=MockModelAdapter(["r1", "r2"]),
                    config=_config(), run_id="bad", command="pytest")
    except ValueError as exc:
        assert "all or nothing" in str(exc)
    else:
        raise AssertionError("expected ValueError")
