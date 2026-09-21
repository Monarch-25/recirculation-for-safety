"""Deterministic mock model adapter for offline tests.

Returns predefined outputs indexed by generation order so the
evaluator, batching, and artifact logic can be tested without
downloading any model.
"""

from __future__ import annotations

from typing import Any, Sequence

from eval_harness.core.interfaces import GenerationConfig, ModelAdapter


class MockModelAdapter(ModelAdapter):
    """Deterministic fake adapter.

    Args:
        outputs: fixed outputs; the i-th generate() call consumes the
            next ``len(prompts)`` entries in order. If ``cycle`` is True,
            outputs wrap around; otherwise exhaustion raises.
        model_id: identifier reported in metadata.
    """

    def __init__(
        self,
        outputs: Sequence[str],
        model_id: str = "mock/mock-model",
        *,
        cycle: bool = False,
    ) -> None:
        if not outputs:
            raise ValueError("MockModelAdapter requires at least one output")
        self._outputs = list(outputs)
        self._model_id = model_id
        self._cycle = cycle
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return "mock-revision"

    @property
    def tokenizer_revision(self) -> str | None:
        return "mock-tokenizer-revision"

    def get_metadata(self) -> dict[str, Any]:
        base = super().get_metadata()
        base.update({"dtype": "mock", "device": "cpu"})
        return base

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        self.calls.append({"num_prompts": len(prompts)})
        out: list[str] = []
        for _ in prompts:
            if self._cursor >= len(self._outputs):
                if self._cycle:
                    self._cursor = 0
                else:
                    raise RuntimeError("MockModelAdapter outputs exhausted")
            out.append(self._outputs[self._cursor])
            self._cursor += 1
        return out

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()
