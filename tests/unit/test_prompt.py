"""Unit tests: prompt rendering + hashing."""

import pytest

from eval_harness.prompting.chat import GSM8K_COT_V1, PromptTemplate, get_template


def test_render_contains_question():
    t = get_template("gsm8k_cot_v1", "1.0")
    out = t.render(question="What is 1+1?")
    assert "What is 1+1?" in out
    assert out == GSM8K_COT_V1.format(question="What is 1+1?")


def test_render_deterministic():
    t = get_template("gsm8k_cot_v1", "1.0")
    assert t.render(question="q") == t.render(question="q")


def test_hash_stable_and_sensitive():
    a = PromptTemplate(name="n", version="1.0", template="hello {x}")
    b = PromptTemplate(name="n", version="1.0", template="hello {x}")
    c = PromptTemplate(name="n", version="1.0", template="hello {x} ")
    assert a.hash == b.hash
    assert a.hash != c.hash
    assert len(a.hash) == 64


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="Unknown prompt template"):
        get_template("nope", "9.9")
