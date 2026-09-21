"""Prompt-template abstraction, separated from tasks and models.

The task selects a template by name; the rendered prompt string is
saved per example. Changing the template changes the experiment
identity (name + version + hash recorded in the manifest).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_harness.utils.hashing import sha256_hex

log = logging.getLogger(__name__)

# Frozen baseline protocol for GSM8K. Do not modify in place —
# create a new versioned template instead.
GSM8K_COT_V1 = """Solve the following math problem step by step.

{question}

Give the final answer clearly."""

# Kojima-style zero-shot CoT (paper-following wording for PT models).
# This is SINGLE-STAGE: one generation + regex answer parsing, unlike
# Kojima et al. (2022) two-stage prompting with a second
# "Therefore, the answer (arabic numerals) is" extraction call.
# Label comparisons honestly: qualitative replication, not exact
# reproduction (see docs/evaluation_protocol.md).
GSM8K_KOJIMA_V1 = "Q: {question} A: Let's think step by step."

# Answer-extraction trigger for Kojima-style two-stage prompting.
# Stage 2 input is "[stage1 prompt] [stage-1 reasoning] [this trigger]",
# exactly the Kojima et al. (2022) "[X'] [Z] [A]" composition for
# numerical answers.
GSM8K_ANSWER_EXTRACT_V1 = "Therefore, the answer (arabic numerals) is"

TEMPLATES: dict[tuple[str, str], str] = {
    ("gsm8k_cot_v1", "1.0"): GSM8K_COT_V1,
    ("gsm8k_kojima_v1", "1.0"): GSM8K_KOJIMA_V1,
    ("gsm8k_answer_extract_v1", "1.0"): GSM8K_ANSWER_EXTRACT_V1,
}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str

    def render(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Missing placeholder {exc} for template '{self.name}'"
            ) from exc

    @property
    def hash(self) -> str:
        return sha256_hex(f"{self.name}\n{self.version}\n{self.template}")


def get_template(name: str, version: str) -> PromptTemplate:
    key = (name, version)
    if key not in TEMPLATES:
        raise ValueError(f"Unknown prompt template {name!r} version {version!r}")
    return PromptTemplate(name=name, version=version, template=TEMPLATES[key])


def format_for_chat(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool,
) -> str:
    """Apply the tokenizer's chat template identically for every backend.

    Both the HF and vLLM adapters use this helper so instruct models see
    byte-identical prompts regardless of inference backend. Falls back to
    the raw prompt when disabled or when the tokenizer has no template.
    """
    if use_chat_template:
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                log.warning("chat template failed, using raw prompt: %s", exc)
    return prompt


def check_bos_present(tokenizer: Any, input_ids: Any, context: str = "",
                      mask: Any = None) -> bool:
    """Guard against the paper-v2 BOS confound.

    Gemma expects a BOS token at the start of every input window; a
    missing BOS inflates baseline perplexity and distorts intervention
    effects. With a padding ``mask``, each row's first REAL token is
    checked (left-padding pads are skipped). Returns True when BOS
    handling looks correct (or the tokenizer defines no BOS token),
    False + warning otherwise. Never raises.
    """
    try:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is None:
            return True

        def _row_ids(i: int):
            row = input_ids[i]
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            if mask is not None:
                m = mask[i]
                m = m.tolist() if hasattr(m, "tolist") else list(m)
                row = [t for t, keep in zip(row, m) if keep]
            return row

        try:
            n_rows = len(input_ids)
        except TypeError:
            return True
        bad = 0
        for i in range(n_rows):
            try:
                row = _row_ids(i)
            except (IndexError, TypeError, ValueError):
                return True  # uninspectable layout; don't cry wolf
            if not row:
                continue
            try:
                first_id = int(row[0])
            except (TypeError, ValueError):
                return True
            if first_id != int(bos_id):
                bad += 1
                if bad <= 4:
                    log.warning(
                        "row %d first real token id=%s is not BOS id=%s "
                        "%s; Gemma-family results without BOS are unreliable",
                        i, first_id, bos_id,
                        f"({context})" if context else "",
                    )
        if bad:
            log.warning("BOS missing on %d/%d rows %s", bad, n_rows,
                        f"({context})" if context else "")
            return False
        return True
    except Exception as exc:
        log.warning("BOS check failed (%s); continuing", exc)
        return True
