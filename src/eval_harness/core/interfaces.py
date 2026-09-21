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

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
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
        }


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
    """

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
