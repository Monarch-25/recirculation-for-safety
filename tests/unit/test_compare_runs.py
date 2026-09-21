"""Unit tests: compare_runs analysis (pure functions, offline)."""

import importlib.util
from pathlib import Path

import pytest


def _load():
    p = Path(__file__).parents[2] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("compare_runs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load()


def _pred(eid, correct, parsed="1", raw="out #### 1", toks=10):
    return {"example_id": eid, "question": "q?", "reference_answer": "#### 1",
            "prompt": "p", "raw_output": raw, "parsed_answer": parsed,
            "parse_success": parsed is not None, "correct": correct,
            "output_tokens": toks}


def test_transition_cells():
    pairs = [( _pred("a", True), _pred("a", True)),
             (_pred("b", True), _pred("b", False)),
             (_pred("c", False), _pred("c", True)),
             (_pred("d", False), _pred("d", False))]
    assert cr.transition_cells(pairs) == {"cc": 1, "cw": 1, "wc": 1, "ww": 1}


def test_pair_partial_overlap():
    a = [_pred("x", True), _pred("y", True)]
    b = [_pred("y", False), _pred("z", True)]
    pairs, a_only, b_only = cr.pair_predictions(a, b)
    assert [p[0]["example_id"] for p in pairs] == ["y"]
    assert a_only == ["x"] and b_only == ["z"]


def test_changed_rows_only_differences():
    pairs = [(_pred("a", True), _pred("a", True)),
             (_pred("b", True, "1", "r1"), _pred("b", False, "2", "r2"))]
    rows = cr.changed_rows(pairs)
    assert [r["example_id"] for r in rows] == ["b"]
    assert rows[0]["baseline_correct"] is True
    assert rows[0]["treatment_correct"] is False
    assert rows[0]["treatment_parsed_answer"] == "2"


def test_summarize_deltas():
    run = lambda acc: {"dir": "d", "manifest": {}, "metrics": {},
                       "preds": []}
    base = run(0)
    treat = run(0)
    base["preds"] = [_pred("a", True), _pred("b", False)]
    treat["preds"] = [_pred("a", False), _pred("b", True)]
    s = cr.summarize(base, treat)
    p = s["paired"]
    assert p["cells"] == {"cc": 0, "cw": 1, "wc": 1, "ww": 0}
    assert p["baseline_accuracy"] == 0.5
    assert p["absolute_delta"] == 0.0
    assert p["relative_delta"] == 0.0
    assert p["rescued_examples"] == 1 and p["regressed_examples"] == 1


def test_relative_delta_none_when_baseline_zero():
    base = {"dir": "d", "manifest": {}, "metrics": {},
            "preds": [_pred("a", False)]}
    treat = {"dir": "e", "manifest": {}, "metrics": {},
             "preds": [_pred("a", True)]}
    s = cr.summarize(base, treat)
    assert s["paired"]["absolute_delta"] == 1.0
    assert s["paired"]["relative_delta"] is None


def test_summarize_no_overlap_raises():
    base = {"dir": "d", "manifest": {}, "metrics": {},
            "preds": [_pred("a", True)]}
    treat = {"dir": "e", "manifest": {}, "metrics": {},
             "preds": [_pred("b", True)]}
    with pytest.raises(ValueError, match="no common"):
        cr.summarize(base, treat)


def test_length_stats_empty_safe():
    assert cr.length_stats([])["chars"]["mean"] is None
    stats = cr.length_stats([_pred("a", True, raw="hello world", toks=None)])
    assert stats["chars"]["mean"] == 11.0
    assert stats["words"]["mean"] == 2.0
    assert stats["tokens"]["mean"] is None
    assert stats["token_coverage"] == 0.0
