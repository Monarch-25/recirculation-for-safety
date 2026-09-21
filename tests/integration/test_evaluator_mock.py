"""Integration: dataset -> evaluator -> mock adapter -> parser -> scorer -> artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.core.config import load_config_from_dict
from eval_harness.core.evaluator import Evaluator
from eval_harness.core.interfaces import EvalExample, ParsedAnswer, Task
from eval_harness.models.mock import MockModelAdapter
from eval_harness.parsing.gsm8k import GSM8KAnswerParser
from eval_harness.prompting.chat import get_template
from eval_harness.scoring.gsm8k import GSM8KScorer
from eval_harness.tasks.gsm8k import make_example_id
from eval_harness.utils.io import read_jsonl

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gsm8k_tiny.json"


class ListTask(Task):
    """In-memory GSM8K-like task (same semantics, offline fixture)."""

    name = "gsm8k"
    version = "1.0"

    def __init__(self, rows, template_name="gsm8k_cot_v1",
                 template_version="1.0"):
        self._rows = rows
        self._template = get_template(template_name, template_version)
        self._parser = GSM8KAnswerParser()
        self._scorer = GSM8KScorer()
        self.dataset_revision = "fixture"

    def load(self, split="test", limit=None):
        rows = self._rows[:limit] if limit else self._rows
        return [EvalExample(
            example_id=make_example_id(i, r["question"]),
            index=i, question=r["question"], reference_answer=r["answer"])
            for i, r in enumerate(rows)]

    def build_prompt(self, example):
        return self._template.render(question=example.question)

    def parse_answer(self, output):
        return self._parser.parse(output)

    def parse_reference(self, example):
        return self._parser.parse_reference(example.reference_answer)

    def score(self, prediction, example):
        return self._scorer.score(prediction, example)


def _rows():
    return json.loads(FIXTURE.read_text())


def _config(batch_size=1, limit=None):
    return load_config_from_dict({
        "model": {"name": "mock/m", "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test", "limit": limit},
        "prompt": {}, "generation": {"max_new_tokens": 32},
        "runtime": {"batch_size": batch_size, "output_dir": "results"},
        "experiment": {"name": "test"},
    })


def _run(rows, outputs, batch_size, tmp_path, run_id=None):
    import uuid
    task = ListTask(rows)
    model = MockModelAdapter(outputs)
    cfg = _config(batch_size)
    ev = Evaluator(output_root=tmp_path / "results")
    return ev.evaluate(task=task, model=model, config=cfg,
                       run_id=run_id or f"test_{uuid.uuid4().hex[:8]}",
                       command="pytest")


def test_full_pipeline_artifacts(tmp_path):
    rows = _rows()[:3]
    outputs = ["answer #### 72", "answer #### 10", "answer #### 5"]
    result = _run(rows, outputs, 1, tmp_path)
    assert result.metrics.num_examples == 3
    assert result.metrics.num_correct == 3
    run_dir = Path(result.run_dir)
    for fname in ("manifest.json", "config.yaml", "metrics.json",
                  "predictions.jsonl", "environment.json", "logs.txt"):
        assert (run_dir / fname).exists(), fname
    preds = read_jsonl(run_dir / "predictions.jsonl")
    assert len(preds) == 3
    for p in preds:
        for key in ("example_id", "prompt", "raw_output", "parsed_answer",
                    "reference_answer", "parse_success", "correct"):
            assert key in p, key


def test_ordering_preserved(tmp_path):
    rows = _rows()[:4]
    outputs = [f"result #### {v}" for v in (72, 10, 5, 280)]
    result = _run(rows, outputs, 2, tmp_path)
    got = [r.parsed_answer for r in result.records]
    assert got == ["72", "10", "5", "280"]
    assert [r.example_id for r in result.records] == \
        [r.example_id for r in ListTask(rows).load("test")]


def test_batch_size_invariance(tmp_path):
    """batch_size must not change example_id -> output mapping."""
    rows = _rows()[:4]
    outputs = [f"result #### {v}" for v in (72, 10, 5, 280)]
    maps = {}
    for bs in (1, 2, 4):
        task = ListTask(rows)
        model = MockModelAdapter(list(outputs))
        cfg = _config(bs)
        ev = Evaluator(output_root=tmp_path / f"r{bs}")
        res = ev.evaluate(task=task, model=model, config=cfg,
                          run_id=f"bs{bs}", command="pytest")
        maps[bs] = {r.example_id: (r.raw_output, r.parsed_answer,
                                   r.correct) for r in res.records}
    assert maps[1] == maps[2] == maps[4]


def test_determinism_twice_identical(tmp_path):
    rows = _rows()[:3]
    outputs = ["a #### 72", "b #### 10", "c #### 5"]
    first = _run(rows, list(outputs), 1, tmp_path / "a")
    second = _run(rows, list(outputs), 1, tmp_path / "b")
    assert [(r.example_id, r.raw_output, r.parsed_answer, r.correct)
            for r in first.records] == \
           [(r.example_id, r.raw_output, r.parsed_answer, r.correct)
            for r in second.records]


def test_parse_failure_does_not_crash(tmp_path):
    rows = _rows()[:2]
    result = _run(rows, ["no numbers at all!", "#### 10"], 1, tmp_path)
    assert result.records[0].parse_success is False
    assert result.records[0].correct is False
    assert result.metrics.num_examples == 2
    assert result.metrics.num_correct == 1


def test_run_dirs_unique(tmp_path):
    rows = _rows()[:2]
    outs = ["#### 72", "#### 10"]
    r1 = _run(rows, list(outs), 1, tmp_path)
    r2 = _run(rows, list(outs), 1, tmp_path)
    # Same output_root but distinct run_ids -> distinct dirs.
    assert r1.run_dir != r2.run_dir
