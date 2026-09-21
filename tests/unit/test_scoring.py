"""Unit tests: GSM8K scoring."""

from eval_harness.core.interfaces import EvalExample, ParsedAnswer
from eval_harness.scoring.gsm8k import GSM8KScorer

S = GSM8KScorer()


def _ex(ref="#### 42"):
    return EvalExample(example_id="t", index=0, question="q",
                       reference_answer=ref)


def test_exact_match():
    assert S.score(ParsedAnswer("42", True), _ex()) == 1.0


def test_mismatch():
    assert S.score(ParsedAnswer("43", True), _ex()) == 0.0


def test_parse_failure_scores_zero():
    assert S.score(ParsedAnswer(None, False), _ex()) == 0.0


def test_numeric_equivalence():
    assert S.score(ParsedAnswer("42.0", True), _ex()) == 1.0


def test_comma_equivalence():
    assert S.score(ParsedAnswer("1000", True),
                   _ex("#### 1,000")) == 1.0
