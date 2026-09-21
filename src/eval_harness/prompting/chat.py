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

TEMPLATES: dict[tuple[str, str], str] = {
    ("gsm8k_cot_v1", "1.0"): GSM8K_COT_V1,
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
