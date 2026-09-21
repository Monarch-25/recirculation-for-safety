"""Canonical interfaces for the evaluation harness.

This module defines the central contracts:

- ``GenerationConfig``: immutable decoding configuration.
- ``ModelAdapter``: owns generation; MUST NOT contain benchmark logic.
- ``Task``: owns benchmark semantics; MUST NOT know the model type.
- ``AnswerParser`` / ``ParsedAnswer``: deterministic answer extraction.
- ``Scorer``: task-specific scoring.

The evaluator orchestrates ``Task`` + ``ModelAdapter`` and depends
only on these abstractions, so a future ``RecirculationModelAdapter``
can be swapped in without changing evaluator/benchmark code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    """Immutable decoding configuration."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42
    # Stage-2 (answer extraction) token budget for two-stage protocols.
    # None = same as max_new_tokens. Extraction answers are a handful of
    # tokens; capping them separately avoids generating hundreds of
    # wasted tokens per example (the dominant cost in serial adapters).
    extraction_max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if (self.extraction_max_new_tokens is not None
                and self.extraction_max_new_tokens <= 0):
            raise ValueError("extraction_max_new_tokens must be > 0 or None")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be in [0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.do_sample is False and self.temperature != 0.0:
            # Deterministic greedy decoding conventionally uses temp 0.
            # We allow it but callers should prefer temp=0 for clarity.
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "extraction_max_new_tokens": self.extraction_max_new_tokens,
        }

    def for_extraction(self) -> "GenerationConfig":
        """Copy with the stage-2 token budget applied (no-op if unset)."""
        import dataclasses
        if self.extraction_max_new_tokens is None:
            return self
        return dataclasses.replace(
            self, max_new_tokens=self.extraction_max_new_tokens)


@dataclass
class BatchTiming:
    """Optional per-generate()-call stats reported by model adapters.

    Convention (P2 §32): an adapter MAY set ``self.last_batch_info`` to a
    ``BatchTiming`` at the end of each :meth:`ModelAdapter.generate` call.
    The evaluator consumes it (then clears it) once per batch. Token
    lists align with the ``prompts`` argument order. Every field is
    optional — adapters report only what they can measure honestly and
    leave the rest None (never fabricate; the evaluator records nulls).
    """

    input_tokens: list[int] | None = None
    output_tokens: list[int] | None = None
    prefill_seconds: float | None = None
    generation_seconds: float | None = None


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Abstract model adapter.

    A model adapter owns *generation only*. It must not contain
    benchmark/task-specific logic (no prompt templates, no answer
    parsing, no scoring).

    Future adapters (Recirculation, vLLM, Modal, API) implement this
    same contract, potentially with token-by-token ``forward()``
    internals, while the evaluator remains unaware.

    Adapters MAY publish per-call stats via the ``last_batch_info``
    attribute (a :class:`BatchTiming` or None); see its docstring.
    """

    last_batch_info: BatchTiming | None = None

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Hugging Face repo id or other model identifier."""
        ...

    @property
    def model_revision(self) -> str | None:
        """Resolved commit hash, or user-pinned revision, or None."""
        return None

    @property
    def tokenizer_revision(self) -> str | None:
        return None

    @property
    def intervention_type(self) -> str:
        """e.g. 'none' for baseline; 'recirculation' for future work."""
        return "none"

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        """Generate one continuation per prompt, preserving order."""
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Metadata for the run manifest. Adapters may extend."""
        return {
            "name": self.model_id,
            "revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "intervention_type": self.intervention_type,
        }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalExample:
    """A single benchmark example with a stable identifier."""

    example_id: str
    index: int
    question: str
    reference_answer: str
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParsedAnswer:
    """Result of deterministic answer parsing."""

    value: str | None
    success: bool


class AnswerParser(ABC):
    """Deterministic answer extractor."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def parse(self, text: str) -> ParsedAnswer: ...

    @abstractmethod
    def parse_reference(self, reference: str) -> ParsedAnswer: ...


class Scorer(ABC):
    """Task-specific scorer mapping (prediction, example) -> score."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def score(self, prediction: ParsedAnswer, example: EvalExample) -> float: ...


class Task(ABC):
    """Abstract benchmark task.

    The task owns dataset loading, prompt construction, answer
    extraction, and scoring. It MUST NOT know whether the model is a
    baseline, Recirculation, or any future intervention.

    Two-stage protocols (e.g. Kojima reasoning + answer extraction) are
    supported via an OPTIONAL method::

        def build_extraction_prompt(
            self, example: EvalExample, reasoning: str) -> str | None: ...

    Tasks without this method (checked via getattr) run single-stage.
    Tasks with it return None to stay single-stage, or a second-stage
    prompt string per example. The evaluator runs both stages through
    the same model adapter and generation config, scores the FINAL
    output, and records the stage-1 reasoning alongside.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def load(
        self,
        split: str,
        limit: int | None = None,
    ) -> list[EvalExample]:
        """Load examples in stable order. ``limit=None`` means full split."""
        ...

    @abstractmethod
    def build_prompt(self, example: EvalExample) -> str:
        """Build a deterministic prompt string for one example."""
        ...

    @abstractmethod
    def parse_answer(self, output: str) -> ParsedAnswer:
        ...

    @abstractmethod
    def parse_reference(self, example: EvalExample) -> ParsedAnswer:
        ...

    @abstractmethod
    def score(self, prediction: ParsedAnswer, example: EvalExample) -> float:
        ...
