"""Integration: two mock runs -> compare_runs artifacts (offline)."""

import importlib.util
import json
from pathlib import Path

import pytest

from eval_harness.core.config import load_config_from_dict
from eval_harness.core.evaluator import Evaluator
from eval_harness.models.mock import MockModelAdapter
from eval_harness.parsing.gsm8k import GSM8KAnswerParser
from tests.integration.test_evaluator_mock import ListTask

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gsm8k_tiny.json"


def _load_compare():
    p = Path(__file__).parents[2] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("compare_runs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    return json.loads(FIXTURE.read_text())[:n]


def _correct_outputs(rows):
    parser = GSM8KAnswerParser()
    return [f"reasoning #### {parser.parse_reference(r['answer']).value}"
            for r in rows]


def _run(rows, outputs, tmp_path, tag):
    task = ListTask(rows)
    model = MockModelAdapter(outputs)
    cfg = load_config_from_dict({
        "model": {"name": "mock/m", "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test"},
        "prompt": {}, "generation": {"max_new_tokens": 32},
        "runtime": {"batch_size": 2, "output_dir": "results"},
        "experiment": {"name": tag},
    })
    ev = Evaluator(output_root=tmp_path / "results")
    return ev.evaluate(task=task, model=model, config=cfg,
                       run_id=f"{tag}", command="pytest")


def test_compare_mock_runs(tmp_path):
    cr = _load_compare()
    rows = _rows(3)
    good = _correct_outputs(rows)
    bad = ["no numbers here!"] + good[1:]
    run_a = _run(rows, good, tmp_path / "a", "baseline")
    run_b = _run(rows, bad, tmp_path / "b", "treatment")
    out = tmp_path / "cmp"
    result = cr.write_comparison(run_a.run_dir, run_b.run_dir, out)
    for fname in ("comparison.json", "comparison.csv",
                  "summary.json", "report.txt"):
        assert (out / fname).exists(), fname
    summary = result["summary"]
    assert summary["paired"]["cells"] == {"cc": 2, "cw": 1,
                                          "wc": 0, "ww": 0}
    assert summary["paired"]["absolute_delta"] == pytest.approx(-1 / 3)
    assert result["n_changed"] == 1
    rows_out = json.loads((out / "comparison.json").read_text())
    assert len(rows_out) == 1
    assert rows_out[0]["baseline_correct"] is True
    assert rows_out[0]["treatment_correct"] is False
    assert "delta=" in result["report"]
