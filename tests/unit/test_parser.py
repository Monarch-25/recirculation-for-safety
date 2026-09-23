"""Unit tests: GSM8K answer parsing."""

from eval_harness.parsing.gsm8k import GSM8KAnswerParser, normalize_number

P = GSM8KAnswerParser()


def test_parse_reference():
    r = P.parse_reference("some reasoning\n#### 72")
    assert (r.value, r.success) == ("72", True)


def test_parse_reference_commas():
    r = P.parse_reference("#### 1,000")
    assert r.value == "1000"


def test_parse_explicit_marker():
    r = P.parse("blah blah #### 42")
    assert (r.value, r.success) == ("42", True)


def test_parse_boxed():
    r = P.parse("The answer is \\boxed{17} done")
    assert (r.value, r.success) == ("17", True)


def test_parse_last_number():
    r = P.parse("First I got 5, then corrected to 7. The answer is 7.")
    assert (r.value, r.success) == ("7", True)


def test_parse_empty_fails():
    assert P.parse("").success is False
    assert P.parse("no numbers here!").success is False


def test_parse_negative_and_decimal():
    r = P.parse("result: -3.5")
    assert r.value == "-3.5"


def test_normalize():
    assert normalize_number(" $1,234.00 ") == "1234.00"
    assert normalize_number("42.") == "42"


def test_parser_versioned():
    assert P.name == "gsm8k_parser"
    assert P.version == "1.0"


def test_parser_v11_answer_line_priority():
    from eval_harness.parsing.gsm8k import GSM8KAnswerParserV11 as P11
    p = P11()
    assert p.version == "1.1"
    # Trailing ramble with other numbers must not override the answer line.
    r = p.parse("The final answer is 6\nTranslated: 6 in Hindi 6.")
    assert (r.value, r.success) == ("6", True)
    r = p.parse("Reasoning 21 - 15 = 6. The final answer is 6\nQ: next")
    assert r.value == "6"


def test_parser_v11_canonicalizes():
    from eval_harness.parsing.gsm8k import GSM8KAnswerParserV11 as P11
    p = P11()
    assert p.parse("The final answer is 1,000.").value == "1000"
    assert p.parse("The final answer is 42.0").value == "42"
    assert p.parse("#### 18").value == "18"
