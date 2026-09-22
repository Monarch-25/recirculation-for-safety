"""Integration: reasoning_partial.jsonl streaming + resume-from-dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.core.evaluator import Evaluator
from eval_harness.models.mock import MockModelAdapter
from eval_harness.utils.io import read_jsonl
from tests.integration.test_evaluator_mock import ListTask, _config, _rows


def _run(rows, outputs, tmp_path, run_id, resume_from=None):
    task = ListTask(rows)
    model = MockModelAdapter(list(outputs))
    cfg = _config(1)
    ev = Evaluator(output_root=tmp_path / "results")
    return ev.evaluate(task=task, model=model, config=cfg,
                       run_id=run_id, command="pytest",
                       resume_from=resume_from)


def test_partial_streamed_with_ids(tmp_path):
    rows = _rows()[:3]
    outs = ["answer #### 72", "answer #### 10", "answer #### 5"]
    res = _run(rows, outs, tmp_path, "s1")
    partial = Path(res.run_dir) / "reasoning_partial.jsonl"
    assert partial.exists()
    lines = read_jsonl(partial)
    assert len(lines) == 3
    assert [l["index"] for l in lines] == [0, 1, 2]
    assert [l["output"] for l in lines] == outs
    ids = [r.example_id for r in res.records]
    assert [l["example_id"] for l in lines] == ids


def test_resume_reproduces_records(tmp_path):
    rows = _rows()[:3]
    outs = ["answer #### 72", "answer #### 10", "answer #### 5"]
    first = _run(rows, outs, tmp_path, "r1")
    # Fresh mock whose outputs must go unused (stage 0 skipped).
    second = _run(rows, ["UNUSED"] * 3, tmp_path, "r2",
                  resume_from=first.run_dir)
    a = [(r.example_id, r.raw_output, r.parsed_answer, r.correct)
         for r in first.records]
    b = [(r.example_id, r.raw_output, r.parsed_answer, r.correct)
         for r in second.records]
    assert a == b
    assert second.metrics.num_correct == first.metrics.num_correct == 3
    manifest = json.loads((Path(second.run_dir) / "manifest.json").read_text())
    assert manifest["resumed_from"] == first.run_dir


def test_resume_rejects_misaligned(tmp_path):
    rows = _rows()[:3]
    outs = ["answer #### 72", "answer #### 10", "answer #### 5"]
    first = _run(rows, outs, tmp_path, "m1")
    bad = tmp_path / "baddir"
    bad.mkdir()
    lines = read_jsonl(Path(first.run_dir) / "reasoning_partial.jsonl")
    (bad / "reasoning_partial.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines[:2]) + "\n")
    with pytest.raises(ValueError, match="2 lines for 3 examples"):
        _run(rows, outs, tmp_path, "m2", resume_from=str(bad))


def test_resume_rejects_missing_file(tmp_path):
    rows = _rows()[:2]
    outs = ["#### 72", "#### 10"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no reasoning_partial.jsonl"):
        _run(rows, outs, tmp_path, "m3", resume_from=str(empty))
