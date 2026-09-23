"""Unit tests: prompt rendering + hashing."""

import pytest

from eval_harness.prompting.chat import (
    GSM8K_COT_LLAMA_FEWSHOTS,
    GSM8K_COT_LLAMA_V1,
    GSM8K_COT_V1,
    GSM8K_KOJIMA_V1,
    GSM8K_ANSWER_EXTRACT_V1,
    PromptTemplate,
    check_bos_present,
    get_template,
)


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


def test_kojima_template_kojima_wording():
    t = get_template("gsm8k_kojima_v1", "1.0")
    out = t.render(question="What is 1+1?")
    assert out == GSM8K_KOJIMA_V1.format(question="What is 1+1?")
    assert "Let's think step by step" in out
    assert out.startswith("Q: ")


class _FakeTok:
    bos_token_id = 2


def test_bos_check_ok():
    assert check_bos_present(_FakeTok(), [[2, 10, 20]]) is True


def test_bos_check_missing_warns():
    assert check_bos_present(_FakeTok(), [[10, 20]]) is False


def test_bos_check_no_bos_token_ok():
    assert check_bos_present(object(), [[10]]) is True


def test_extract_trigger_template():
    t = get_template("gsm8k_answer_extract_v1", "1.0")
    assert t.render() == GSM8K_ANSWER_EXTRACT_V1
    assert "arabic numerals" in t.render()


# --- gsm8k_cot_llama_v1 (Evalution cot_llama parity) -----------------


def test_cot_llama_bare_query_block():
    # The single-query block (their `llama_prompt`) with NO fewshots.
    bare = PromptTemplate(name="gsm8k_cot_llama_v1", version="1.0",
                          template=GSM8K_COT_LLAMA_V1)
    out = bare.render(question="What is 1+1?")
    assert "Given the following problem" in out
    assert "Problem: What is 1+1?" in out
    assert 'The final answer is [answer]' in out
    assert out == GSM8K_COT_LLAMA_V1.format(question="What is 1+1?")


def test_cot_llama_fewshot_pool_wired():
    t = get_template("gsm8k_cot_llama_v1", "1.0")
    assert len(t.fewshots) == 8
    assert all("question" in fs and "target" in fs for fs in t.fewshots)
    # Byte-identical to Evalution's _LLAMA_FEWSHOTS count + wording.
    assert t.fewshots[0]["target"].endswith("The final answer is 6")
    assert t.fewshots[7]["target"].endswith("The final answer is 8")
    assert len(GSM8K_COT_LLAMA_FEWSHOTS) == 8


def test_cot_llama_fewshot_join_layout():
    t = get_template("gsm8k_cot_llama_v1", "1.0")
    out = t.render(question="The query")
    # Evalution _build_plain_prompt layout: (fs prompt + " " + target +
    # "\n\n") * 8, then the query prompt.
    parts = []
    for fs in t.fewshots:
        parts.append(GSM8K_COT_LLAMA_V1.format(question=fs["question"]))
        parts.append(" ")
        parts.append(fs["target"])
        parts.append("\n\n")
    parts.append(GSM8K_COT_LLAMA_V1.format(question="The query"))
    assert out == "".join(parts)
    assert out.count("Given the following problem") == 9
    assert out.endswith(GSM8K_COT_LLAMA_V1.format(question="The query"))


def test_cot_llama_hash_differs_from_cot_v1():
    assert (get_template("gsm8k_cot_llama_v1", "1.0").hash
            != get_template("gsm8k_cot_v1", "1.0").hash)


def test_cot_llama_multiturn_message_layout():
    from eval_harness.prompting import chat as C
    t = C.get_template("gsm8k_cot_llama_multiturn_v1", "1.0")
    assert t.multiturn is True
    msgs = t.render_messages(question="The query")
    # 8 fewshot (user, assistant) turns + final query user turn.
    assert len(msgs) == 17
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[-1] == {
        "role": "user",
        "content": C.GSM8K_COT_LLAMA_V1.format(question="The query"),
    }
    assert msgs[1]["content"].endswith("The final answer is 6")
    # Single-turn render must refuse multiturn templates.
    try:
        t.render(question="x")
    except ValueError:
        pass
    else:
        raise AssertionError("render() should refuse multiturn templates")


def test_multiturn_hash_differs_from_single_turn():
    from eval_harness.prompting import chat as C
    assert (C.get_template("gsm8k_cot_llama_multiturn_v1", "1.0").hash
            != C.get_template("gsm8k_cot_llama_v1", "1.0").hash)


def test_single_turn_hash_frozen():
    # v1.0 single-turn hashes keep the original formula (no suffix).
    from eval_harness.prompting import chat as C
    from eval_harness.utils.hashing import sha256_hex
    t = C.get_template("gsm8k_cot_llama_v1", "1.0")
    fs = "".join(f"{f['question']} => {f['target']}" for f in t.fewshots)
    assert t.hash == sha256_hex(
        f"{t.name}\n{t.version}\n{t.template}\n{fs}")
