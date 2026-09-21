"""GSM8K task: dataset loading, prompting, parsing, scoring.

Dataset: ``openai/gsm8k`` config ``main`` (fallback: ``gsm8k``), test
split. ``limit=None`` means the full split (1319 examples).

Stable example IDs: the source dataset provides no id field, so ids
are ``gsm8k-test-{index:05d}-{sha8(question)}`` — deterministic in the
dataset order (index) and content-addressed (hash). Order is the raw
dataset order, truncated to ``limit`` when set. Bump ``version`` when
benchmark semantics change.
"""

from __future__ import annotations

import logging
from typing import Any

from eval_harness.core.interfaces import EvalExample, ParsedAnswer, Task
from eval_harness.parsing.gsm8k import GSM8KAnswerParser
from eval_harness.prompting.chat import get_template
from eval_harness.scoring.gsm8k import GSM8KScorer
from eval_harness.utils.hashing import sha256_hex
log = logging.getLogger(__name__)

PRIMARY_DATASET_ID = "openai/gsm8k"
FALLBACK_DATASET_ID = "gsm8k"
DATASET_CONFIG = "main"


def make_example_id(index: int, question: str, split: str = "test") -> str:
    digest = sha256_hex(question)[:8]
    return f"gsm8k-{split}-{index:05d}-{digest}"


class GSM8KTask(Task):
    name = "gsm8k"
    version = "1.0"

    def __init__(
        self,
        template_name: str = "gsm8k_cot_v1",
        template_version: str = "1.0",
        dataset_id: str = PRIMARY_DATASET_ID,
        dataset_config: str = DATASET_CONFIG,
        dataset_revision: str | None = None,
        extraction_template_name: str | None = None,
        extraction_template_version: str = "1.0",
    ) -> None:
        self._template = get_template(template_name, template_version)
        self._parser = GSM8KAnswerParser()
        self._scorer = GSM8KScorer()
        self._dataset_id = dataset_id
        self._dataset_config = dataset_config
        self._dataset_revision = dataset_revision
        self._resolved_dataset_id = dataset_id
        self._resolved_dataset_revision = dataset_revision
        self.extraction_template_name = extraction_template_name
        self.extraction_template_version = extraction_template_version
        self._extract_template = (
            get_template(extraction_template_name,
                         extraction_template_version)
            if extraction_template_name else None)

    @property
    def dataset_revision(self) -> str | None:
        return self._resolved_dataset_revision

    def _load_hf_dataset(self, split: str):
        from datasets import load_dataset

        candidates = [self._dataset_id]
        if self._dataset_id == PRIMARY_DATASET_ID:
            candidates.append(FALLBACK_DATASET_ID)
        last_err: Exception | None = None
        for ds_id in candidates:
            try:
                log.info("loading dataset %s config=%s split=%s",
                         ds_id, self._dataset_config, split)
                if ds_id == FALLBACK_DATASET_ID:
                    ds = load_dataset(ds_id, split=split)
                else:
                    ds = load_dataset(
                        ds_id, self._dataset_config, split=split,
                        revision=self._dataset_revision,
                    )
                self._resolved_dataset_id = ds_id
                self._try_resolve_revision(ds_id)
                return ds
            except Exception as exc:
                log.warning("dataset load failed for %s: %s", ds_id, exc)
                last_err = exc
        raise RuntimeError(f"Could not load GSM8K dataset: {last_err}")

    def _try_resolve_revision(self, ds_id: str) -> None:
        if self._dataset_revision:
            self._resolved_dataset_revision = self._dataset_revision
            return
        try:
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(ds_id)
            self._resolved_dataset_revision = info.sha
        except Exception as exc:
            log.warning("could not resolve dataset revision: %s", exc)
            self._resolved_dataset_revision = None

    def load(self, split: str, limit: int | None = None) -> list[EvalExample]:
        if split not in ("test", "train"):
            raise ValueError(f"Unsupported GSM8K split: {split!r}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0 or None")
        ds = self._load_hf_dataset(split)
        examples: list[EvalExample] = []
        total = len(ds) if limit is None else min(limit, len(ds))
        for i in range(total):
            row: dict[str, Any] = ds[i]
            question = str(row["question"])
            answer = str(row["answer"])
            examples.append(EvalExample(
                example_id=make_example_id(i, question, split),
                index=i,
                question=question,
                reference_answer=answer,
            ))
        log.info("loaded %d/%d GSM8K examples (split=%s)",
                 len(examples), len(ds), split)
        return examples

    def build_prompt(self, example: EvalExample) -> str:
        return self._template.render(question=example.question)

    def build_extraction_prompt(
        self,
        example: EvalExample,
        reasoning: str,
    ) -> str | None:
        """Stage-2 prompt ("[X'] [Z] [A]"), or None for single-stage."""
        if self._extract_template is None:
            return None
        stage1 = self.build_prompt(example)
        trigger = self._extract_template.render()
        return f"{stage1} {reasoning.strip()} {trigger}".strip()

    def parse_answer(self, output: str) -> ParsedAnswer:
        return self._parser.parse(output)

    def parse_reference(self, example: EvalExample) -> ParsedAnswer:
        return self._parser.parse_reference(example.reference_answer)

    def score(self, prediction: ParsedAnswer, example: EvalExample) -> float:
        return self._scorer.score(prediction, example)

    @classmethod
    def from_examples(
        cls,
        rows: list[dict[str, str]],
        template_name: str = "gsm8k_cot_v1",
        template_version: str = "1.0",
        extraction_template_name: str | None = None,
        extraction_template_version: str = "1.0",
    ) -> tuple["GSM8KTask", list[EvalExample]]:
        """Build a task + example list from in-memory rows (tests only)."""
        task = cls(template_name=template_name,
                   template_version=template_version,
                   extraction_template_name=extraction_template_name,
                   extraction_template_version=extraction_template_version)
        examples = [
            EvalExample(
                example_id=make_example_id(i, r["question"]),
                index=i,
                question=r["question"],
                reference_answer=r["answer"],
            )
            for i, r in enumerate(rows)
        ]
        return task, examples
