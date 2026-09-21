"""GSM8K answer parser (deterministic, versioned).

Policy v1.0 ("gsm8k_parser v1.0"):
  1. Reference answers have the form "... reasoning ... #### 42".
     ``parse_reference`` extracts the substring after the LAST "####".
  2. Model outputs are free-form. ``parse`` tries, in order:
     a. substring after the last "####" (if present);
     b. content inside the last ``\\boxed{...}`` (if present);
     c. the last number-like token in the text.
  3. A number-like token matches ``-?\\d[\\d,]*\\.?\\d*``; normalization
     strips whitespace, commas, "$", and surrounding punctuation.
  4. If no candidate is found -> ``ParsedAnswer(None, False)``.

No other heuristics are applied. Any change requires a version bump
and an update to docs/evaluation_protocol.md.
"""

from __future__ import annotations

import re

from eval_harness.core.interfaces import AnswerParser, ParsedAnswer

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def normalize_number(text: str) -> str:
    """Normalize a numeric string for comparison."""
    s = text.strip()
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    s = s.strip(".,;:!?\"'()[]{}")
    return s


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    # Filter out empty / lone "-" artifacts.
    cleaned = [m for m in matches if m not in ("", "-", "-.")]
    if not cleaned:
        return None
    return normalize_number(cleaned[-1]) or None


class GSM8KAnswerParser(AnswerParser):
    name = "gsm8k_parser"
    version = "1.0"

    def parse_reference(self, reference: str) -> ParsedAnswer:
        if "####" in reference:
            tail = reference.rsplit("####", 1)[1].strip()
            num = _last_number(tail)
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        # Fallback: last number in the whole reference string.
        num = _last_number(reference)
        return ParsedAnswer(value=num, success=num is not None)

    def parse(self, text: str) -> ParsedAnswer:
        if not text or not text.strip():
            return ParsedAnswer(value=None, success=False)
        # (a) explicit GSM8K marker
        if "####" in text:
            tail = text.rsplit("####", 1)[1].strip()
            num = _last_number(tail)
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        # (b) \boxed{...}
        boxed = _BOXED_RE.findall(text)
        if boxed:
            num = _last_number(boxed[-1])
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        # (c) last number in the text
        num = _last_number(text)
        return ParsedAnswer(value=num, success=num is not None)
