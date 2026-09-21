"""Result records, aggregation, and manifest schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RESULT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PredictionRecord:
    """Per-example record. One row per example in predictions.jsonl."""

    example_id: str
    index: int
    prompt: str
    raw_output: str
    parsed_answer: str | None
    reference_answer: str
    parsed_reference: str | None
    parse_success: bool
    correct: bool
    score: float
    # Optional per-example token counts (P2 §22/§32). None when the
    # adapter cannot report them (e.g. mock); never fabricated.
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Source question text (P2 §20 per-example diffs). Old runs lack it.
    question: str | None = None
    # Two-stage protocols: stage-1 reasoning + stage-2 prompt. None for
    # single-stage runs. raw_output always holds the SCORED (final) text.
    reasoning: str | None = None
    extraction_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregatedMetrics:
    num_examples: int
    num_correct: int
    accuracy: float
    parse_rate: float
    num_parse_failures: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(records: list[PredictionRecord]) -> AggregatedMetrics:
    n = len(records)
    if n == 0:
        return AggregatedMetrics(
            num_examples=0, num_correct=0, accuracy=0.0,
            parse_rate=0.0, num_parse_failures=0,
        )
    num_correct = sum(1 for r in records if r.correct)
    num_parsed = sum(1 for r in records if r.parse_success)
    return AggregatedMetrics(
        num_examples=n,
        num_correct=num_correct,
        accuracy=num_correct / n,
        parse_rate=num_parsed / n,
        num_parse_failures=n - num_parsed,
    )


@dataclass
class EvaluationResult:
    """Aggregated evaluation outcome plus per-example records."""

    run_id: str
    records: list[PredictionRecord]
    metrics: AggregatedMetrics
    manifest: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "metrics": self.metrics.to_dict(),
            "manifest": self.manifest,
        }
