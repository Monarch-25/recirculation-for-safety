"""Unit tests: vLLM adapter with the `vllm` package stubbed.

These run offline (no GPU, no downloads). A real-backend test lives in
tests/integration/test_vllm_smoke.py under the `vllm` marker.
"""

import sys
import types

import pytest

from eval_harness.core.config import load_config_from_dict
from eval_harness.core.interfaces import GenerationConfig


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert messages[0]["role"] == "user"
        return "<user>" + messages[0]["content"] + "</user>"


class _FakeCompletion:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeCompletion(text)]


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLLM.instances.append(self)

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params):
        self.last_prompts = list(prompts)
        self.last_params = params
        return [_FakeRequestOutput(f"gen:{i}") for i, _ in enumerate(prompts)]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def stub_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr(
        "eval_harness.models.vllm_adapter.resolve_hub_revision",
        lambda repo, pinned, kind="model": pinned or "stub-sha",
    )
    FakeLLM.instances.clear()
    return mod


def _make(stub_vllm, **over):
    from eval_harness.models.vllm_adapter import VLLMModelAdapter
    kwargs = {"dtype": "auto"}
    kwargs.update(over)
    return VLLMModelAdapter("org/model", **kwargs)


def test_greedy_mapping(stub_vllm):
    adapter = _make(stub_vllm)
    cfg = GenerationConfig(max_new_tokens=64, temperature=0.9, top_p=0.5,
                           do_sample=False, seed=7)
    assert adapter.generate(["hello"], cfg) == ["gen:0"]
    params = FakeLLM.instances[-1].last_params.kwargs
    assert params["temperature"] == 0.0
    assert params["top_p"] == 1.0
    assert params["max_tokens"] == 64
    assert params["seed"] == 7
    assert params["n"] == 1


def test_sampling_passthrough(stub_vllm):
    adapter = _make(stub_vllm)
    cfg = GenerationConfig(max_new_tokens=32, temperature=0.7, top_p=0.9,
                           do_sample=True, seed=1)
    adapter.generate(["hello"], cfg)
    params = FakeLLM.instances[-1].last_params.kwargs
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9


def test_chat_template_shared_with_hf(stub_vllm):
    """Prompts must match the HF adapter byte-for-byte for the same template.

    Both adapters delegate to ``prompting.chat.format_for_chat``; the HF
    adapter's ``_format_prompt`` is a thin wrapper around it.
    """
    import inspect

    from eval_harness.models import hf_causal_lm
    from eval_harness.prompting.chat import format_for_chat
    src = inspect.getsource(hf_causal_lm.HFCausalLMAdapter._format_prompt)
    assert "format_for_chat" in src
    adapter = _make(stub_vllm)
    adapter.generate(["What is 2+2?"],
                     GenerationConfig(max_new_tokens=8))
    sent = FakeLLM.instances[-1].last_prompts
    assert sent == [format_for_chat(FakeTokenizer(), "What is 2+2?", True)]


def test_order_preserved(stub_vllm):
    adapter = _make(stub_vllm)
    out = adapter.generate(["a", "b", "c"],
                           GenerationConfig(max_new_tokens=8))
    assert out == ["gen:0", "gen:1", "gen:2"]


def test_empty_prompts(stub_vllm):
    adapter = _make(stub_vllm)
    assert adapter.generate([], GenerationConfig(max_new_tokens=8)) == []


def test_constructor_kwargs(stub_vllm):
    _make(stub_vllm, revision="abc123", tensor_parallel_size=2,
          gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=4096)
    kw = FakeLLM.instances[-1].kwargs
    assert kw["model"] == "org/model"
    assert kw["revision"] == "abc123"
    assert kw["tensor_parallel_size"] == 2
    assert kw["gpu_memory_utilization"] == 0.8
    assert kw["enforce_eager"] is True
    assert kw["max_model_len"] == 4096


@pytest.mark.parametrize("tp,gmu", [(0, 0.9), (1, 0.0), (1, 1.5)])
def test_invalid_engine_args(stub_vllm, tp, gmu):
    from eval_harness.models.vllm_adapter import VLLMModelAdapter
    with pytest.raises(ValueError):
        VLLMModelAdapter("org/model", tensor_parallel_size=tp,
                         gpu_memory_utilization=gmu)


def test_metadata(stub_vllm):
    adapter = _make(stub_vllm, revision="abc123")
    meta = adapter.get_metadata()
    assert meta["backend"] == "vllm"
    assert meta["name"] == "org/model"
    assert meta["revision"] == "abc123"
    assert meta["tokenizer_revision"] == "abc123"
    assert meta["tensor_parallel_size"] == 1
    assert adapter.model_id == "org/model"
    assert adapter.intervention_type == "none"


def test_missing_vllm_raises_helpful_error(monkeypatch):
    from eval_harness.models.vllm_adapter import VLLMModelAdapter
    monkeypatch.setitem(sys.modules, "vllm", None)  # force ImportError
    with pytest.raises(ImportError, match="Linux-only"):
        VLLMModelAdapter("org/model")


def test_config_backend_validation():
    raw = {"model": {"name": "m", "backend": "tensorrt"},
           "task": {"name": "gsm8k"}}
    with pytest.raises(ValueError, match="backend"):
        load_config_from_dict(raw)


def test_config_vllm_knobs():
    raw = {"model": {"name": "m", "backend": "vllm",
                     "tensor_parallel_size": 2,
                     "gpu_memory_utilization": 0.7,
                     "enforce_eager": True,
                     "max_model_len": 2048},
           "task": {"name": "gsm8k"}}
    cfg = load_config_from_dict(raw)
    assert cfg.model.backend == "vllm"
    assert cfg.model.tensor_parallel_size == 2
    assert cfg.model.gpu_memory_utilization == 0.7
    assert cfg.model.enforce_eager is True
    assert cfg.model.max_model_len == 2048
    # Round-trips through serialization (config.yaml in run dirs).
    assert load_config_from_dict(cfg.to_dict()).model.backend == "vllm"
