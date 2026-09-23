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


# ---------------------------------------------------------------------------
# Policy v1.1 ("gsm8k_parser v1.1", ModelCloud/Evalution parity)
# ---------------------------------------------------------------------------
# Priority mirrors Evalution's
# ``extract_format_insensitive_numeric_answer``:
#   1. ``#### N`` marker (first number after the marker);
#   2. LAST "the final answer is / answer is / answer:" line (first
#      numeric token in the line remainder);
#   3. ``\\boxed{...}`` (first numeric token inside);
#   4. last non-empty line's last numeric token;
#   5. last numeric token in the whole text.
# Values are Decimal-canonicalized ("42.0" -> "42", commas stripped) so
# "1,000" and "1000" compare equal. v1.0 is frozen for old-run
# reproducibility; new repo-parity configs select v1.1 explicitly.

_HASH_RE = re.compile(
    r"####\s*(-?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+))")
_ANSWER_LINE_RE = re.compile(
    r"(?:the\s+final\s+answer\s+is|final\s+answer\s*:|the\s+answer\s+is"
    r"|answer\s*:)\s*(.+)",
    re.IGNORECASE)


def canonicalize_numeric_token(token: str) -> str | None:
    """Decimal-canonicalize a numeric token; None when not numeric."""
    from decimal import Decimal, InvalidOperation

    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in ("", "-0"):
        return "0"
    return normalized


def _first_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    for m in matches:
        if m in ("", "-", "-."):
            continue
        canon = canonicalize_numeric_token(m)
        if canon is not None:
            return canon
    return None


class GSM8KAnswerParserV11(AnswerParser):
    name = "gsm8k_parser"
    version = "1.1"

    def parse_reference(self, reference: str) -> ParsedAnswer:
        if "####" in reference:
            tail = reference.rsplit("####", 1)[1].strip()
            num = _first_number(tail)
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        num = _last_number(reference)
        if num is None:
            return ParsedAnswer(value=None, success=False)
        return ParsedAnswer(
            value=canonicalize_numeric_token(num) or num, success=True)

    def parse(self, text: str) -> ParsedAnswer:
        if not text or not text.strip():
            return ParsedAnswer(value=None, success=False)
        hashed = _HASH_RE.search(text)
        if hashed is not None:
            canon = canonicalize_numeric_token(hashed.group(1))
            if canon is not None:
                return ParsedAnswer(value=canon, success=True)
        answer_lines = list(_ANSWER_LINE_RE.finditer(text))
        if answer_lines:
            num = _first_number(answer_lines[-1].group(1))
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        boxed = _BOXED_RE.findall(text)
        if boxed:
            num = _first_number(boxed[-1])
            if num is not None:
                return ParsedAnswer(value=num, success=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            num = _last_number(lines[-1])
            if num is not None:
                canon = canonicalize_numeric_token(num) or num
                return ParsedAnswer(value=canon, success=True)
        num = _last_number(text)
        if num is None:
            return ParsedAnswer(value=None, success=False)
        return ParsedAnswer(
            value=canonicalize_numeric_token(num) or num, success=True)
