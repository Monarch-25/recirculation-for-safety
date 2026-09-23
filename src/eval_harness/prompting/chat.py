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

# Evalution `cot_llama` (ModelCloud/Recirculation) question prompt —
# byte-identical to their `llama_prompt` doc_to_text so replicating runs
# match the repo protocol under `apply_chat_template=false`.
GSM8K_COT_LLAMA_V1 = (
    "Given the following problem, reason and give a final answer to the "
    "problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where '
    "[answer] is the response to the problem.\n"
)

# Fixed 8-shot pool for `cot_llama` — a faithful copy of Evalution's
# `_LLAMA_FEWSHOTS` (question + target ending in "The final answer is N").
# The 8-shot running prompt is built as in Evalution's
# `_build_plain_prompt`:
#     for fs in fewshots:
#         prompt += llama_prompt(fs) + " " + fs.target + "\n\n"
#     prompt += llama_prompt(query)
GSM8K_COT_LLAMA_FEWSHOTS: tuple[dict[str, str], ...] = (
    {
        "question": ("There are 15 trees in the grove. Grove workers will "
                     "plant trees in the grove today. After they are done, "
                     "there will be 21 trees. How many trees did the grove "
                     "workers plant today?"),
        "target": ("There are 15 trees originally. Then there were 21 trees "
                   "after some more were planted. So there must have been "
                   "21 - 15 = 6. The final answer is 6"),
    },
    {
        "question": ("If there are 3 cars in the parking lot and 2 more cars "
                     "arrive, how many cars are in the parking lot?"),
        "target": ("There are originally 3 cars. 2 more cars arrive. 3 + 2 "
                   "= 5. The final answer is 5"),
    },
    {
        "question": ("Leah had 32 chocolates and her sister had 42. If they "
                     "ate 35, how many pieces do they have left in total?"),
        "target": ("Originally, Leah had 32 chocolates. Her sister had 42. "
                   "So in total they had 32 + 42 = 74. After eating 35, "
                   "they had 74 - 35 = 39. The final answer is 39"),
    },
    {
        "question": ("Jason had 20 lollipops. He gave Denny some lollipops. "
                     "Now Jason has 12 lollipops. How many lollipops did "
                     "Jason give to Denny?"),
        "target": ("Jason started with 20 lollipops. Then he had 12 after "
                   "giving some to Denny. So he gave Denny 20 - 12 = 8. The "
                   "final answer is 8"),
    },
    {
        "question": ("Shawn has five toys. For Christmas, he got two toys "
                     "each from his mom and dad. How many toys does he have "
                     "now?"),
        "target": ("Shawn started with 5 toys. If he got 2 toys each from "
                   "his mom and dad, then that is 4 more toys. 5 + 4 = 9. "
                   "The final answer is 9"),
    },
    {
        "question": ("There were nine computers in the server room. Five "
                     "more computers were installed each day, from monday to "
                     "thursday. How many computers are now in the server "
                     "room?"),
        "target": ("There were originally 9 computers. For each of 4 days, "
                   "5 more computers were added. So 5 * 4 = 20 computers "
                   "were added. 9 + 20 is 29. The final answer is 29"),
    },
    {
        "question": ("Michael had 58 golf balls. On tuesday, he lost 23 golf "
                     "balls. On wednesday, he lost 2 more. How many golf "
                     "balls did he have at the end of wednesday?"),
        "target": ("Michael started with 58 golf balls. After losing 23 on "
                   "tuesday, he had 58 - 23 = 35. After losing 2 more, he "
                   "had 35 - 2 = 33 golf balls. The final answer is 33"),
    },
    {
        "question": ("Olivia has $23. She bought five bagels for $3 each. "
                     "How much money does she have left?"),
        "target": ("Olivia had 23 dollars. 5 bagels for 3 dollars each will "
                   "be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. "
                   "23 - 15 is 8. The final answer is 8"),
    },
)

TEMPLATES: dict[tuple[str, str], str] = {
    ("gsm8k_cot_v1", "1.0"): GSM8K_COT_V1,
    ("gsm8k_kojima_v1", "1.0"): GSM8K_KOJIMA_V1,
    ("gsm8k_cot_llama_v1", "1.0"): GSM8K_COT_LLAMA_V1,
    ("gsm8k_answer_extract_v1", "1.0"): GSM8K_ANSWER_EXTRACT_V1,
}

# Templates rendered as multi-turn chat (Evalution `fewshot_as_multiturn`
# with `apply_chat_template=true`): 8 fewshot (user, assistant) turns plus
# the final query user turn, passed as messages to
# `tokenizer.apply_chat_template`. Single-turn `gsm8k_cot_llama_v1` packs
# the same content into one user message and is NOT equivalent.
MULTITURN_TEMPLATES: dict[tuple[str, str], str] = {
    ("gsm8k_cot_llama_multiturn_v1", "1.0"): GSM8K_COT_LLAMA_V1,
}

# Few-shot pools keyed by template name (used when rendering
# `cot_llama`-style prompts exactly like Evalution's `_build_plain_prompt`).
FEWSHOT_POOLS: dict[str, tuple[dict[str, str], ...]] = {
    "gsm8k_cot_llama_v1": GSM8K_COT_LLAMA_FEWSHOTS,
}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    fewshots: tuple[dict[str, str], ...] = ()
    multiturn: bool = False

    def render(self, **kwargs) -> str:
        if self.multiturn:
            raise ValueError(
                f"Template '{self.name}' is multi-turn; use render_messages()"
            )
        if not self.fewshots:
            return self._format(**kwargs)
        # Evalution cot_llama layout: (fs prompt + " " + target + "\n\n")*8
        # then the query prompt, byte-identical to `_build_plain_prompt`.
        parts: list[str] = []
        for fs in self.fewshots:
            parts.append(self._format(question=fs["question"]))
            parts.append(" ")
            parts.append(fs["target"])
            parts.append("\n\n")
        parts.append(self._format(**kwargs))
        return "".join(parts)

    def _format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Missing placeholder {exc} for template '{self.name}'"
            ) from exc

    def render_messages(self, **kwargs) -> list[dict[str, str]]:
        """Evalution `fewshot_as_multiturn` layout (repo-parity).

        One user/assistant turn per fewshot plus the final query user
        turn: ``[U(fs1), A(t1), ..., U(fs8), A(t8), U(query)]``. The
        caller applies the tokenizer chat template with
        ``add_generation_prompt=True``.
        """
        if not self.multiturn:
            raise ValueError(
                f"Template '{self.name}' is single-turn; use render()"
            )
        messages: list[dict[str, str]] = []
        for fs in self.fewshots:
            messages.append({"role": "user",
                             "content": self._format(question=fs["question"])})
            messages.append({"role": "assistant", "content": fs["target"]})
        messages.append({"role": "user", "content": self._format(**kwargs)})
        return messages

    @property
    def hash(self) -> str:
        fs = "".join(f"{f['question']} => {f['target']}"
                      for f in self.fewshots)
        # Frozen: single-turn hashes keep the original formula (no
        # suffix); only multi-turn templates carry the mode marker.
        suffix = "\nmultiturn=True" if self.multiturn else ""
        return sha256_hex(
            f"{self.name}\n{self.version}\n{self.template}\n{fs}{suffix}")


def get_template(name: str, version: str) -> PromptTemplate:
    key = (name, version)
    if key in TEMPLATES:
        return PromptTemplate(
            name=name, version=version, template=TEMPLATES[key],
            fewshots=FEWSHOT_POOLS.get(name, ()), multiturn=False)
    if key in MULTITURN_TEMPLATES:
        return PromptTemplate(
            name=name, version=version, template=MULTITURN_TEMPLATES[key],
            fewshots=FEWSHOT_POOLS.get("gsm8k_cot_llama_v1", ()),
            multiturn=True)
    raise ValueError(f"Unknown prompt template {name!r} version {version!r}")


def format_for_chat(
    tokenizer: Any,
    prompt: str | list[dict[str, str]],
    use_chat_template: bool,
) -> str:
    """Apply the tokenizer's chat template identically for every backend.

    Both the HF and vLLM adapters use this helper so instruct models see
    byte-identical prompts regardless of inference backend. Falls back to
    the raw prompt when disabled or when the tokenizer has no template.

    ``prompt`` may be a pre-built message list (multi-turn templates);
    it is applied with ``add_generation_prompt=True`` exactly as the
    ModelCloud/Evalution ``cot_llama`` + ``apply_chat_template`` path.
    """
    if isinstance(prompt, list):
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                log.warning("chat template failed, using raw prompt: %s", exc)
        return "\n\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt)
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
