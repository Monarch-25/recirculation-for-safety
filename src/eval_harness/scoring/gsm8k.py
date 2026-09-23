"""GSM8K scorer: exact match after normalization.

correct = (normalized prediction == normalized reference), with a
numeric-equivalence fallback (e.g. "42" vs "42.0"). Parse failures
score 0 and are INCLUDED in the accuracy denominator.
"""

from __future__ import annotations

from eval_harness.core.interfaces import EvalExample, ParsedAnswer, Scorer
from eval_harness.parsing.gsm8k import (
    GSM8KAnswerParser,
    normalize_number,
)


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return False


class GSM8KScorer(Scorer):
    name = "gsm8k_exact_match"
    version = "1.0"

    def __init__(self, parser: GSM8KAnswerParser | None = None) -> None:
        self._parser = parser or GSM8KAnswerParser()

    def score(self, prediction: ParsedAnswer, example: EvalExample) -> float:
        if not prediction.success or prediction.value is None:
            return 0.0
        reference = self._parser.parse_reference(example.reference_answer)
        if not reference.success or reference.value is None:
            return 0.0
        pred = normalize_number(prediction.value)
        ref = normalize_number(reference.value)
        if pred == ref:
            return 1.0
        return 1.0 if _numeric_equal(pred, ref) else 0.0
