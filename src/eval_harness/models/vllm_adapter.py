"""vLLM adapter for optimized inference (Linux/CUDA-targeted).

Implements the same :class:`ModelAdapter` contract as
:class:`HFCausalLMAdapter`, so the evaluator, tasks, prompts, parsing,
scoring, and artifact format are unchanged — only this adapter differs.

Decoding parity with the HF baseline:
  - ``do_sample=False`` maps to ``temperature=0.0`` (vLLM greedy),
    ``top_p=1.0``; sampling configs pass through verbatim.
  - Prompts go through the shared :func:`format_for_chat` helper, so
    instruct models see byte-identical prompts on both backends.
  - ``seed`` is passed both to the engine and per-request sampling.

Platform notes:
  - vLLM publishes Linux x86_64 wheels only; it cannot run on Apple
    Silicon macOS. Install with ``pip install -e ".[vllm]"`` on Linux.
  - The import is lazy: constructing this adapter without vLLM
    installed raises an informative ImportError instead of breaking
    ``import eval_harness`` or the offline test suite.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from importlib import metadata
from typing import Any, Sequence

from eval_harness.core.config import (
    InterventionConfig,
    normalize_intervention,
)
from eval_harness.core.interfaces import (
    BatchTiming,
    GenerationConfig,
    ModelAdapter,
)
from eval_harness.prompting.chat import check_bos_present, format_for_chat
from eval_harness.utils.hub import resolve_hub_revision

log = logging.getLogger(__name__)


def _vllm_version() -> str | None:
    try:
        return metadata.version("vllm")
    except Exception:
        return None


class VLLMModelAdapter(ModelAdapter):
    """Baseline vLLM adapter (intervention_type='none')."""

    backend = "vllm"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        seed: int = 0,
        intervention: InterventionConfig | dict[str, Any] | None = None,
    ) -> None:
        # vLLM's v1 engine starts its EngineCore in a forked subprocess.
        # If the parent has already initialized CUDA (e.g. an earlier
        # torch.cuda.is_available() during device resolution), the child
        # dies with "Cannot re-initialize CUDA in forked subprocess".
        # Forcing spawn gives the child a fresh interpreter (documented
        # vLLM workaround). No-op when already spawn or on CUDA-free
        # machines; runs before any vLLM import/construction.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # The v1 sampler defaults to a flashinfer JIT path that needs nvcc
        # at engine warmup; minimal images (Modal debian_slim) ship only
        # the CUDA runtime. Disabling it selects vLLM's torch-native
        # sampler — identical greedy outputs, no compiler needed.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        try:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError as exc:
            log.warning("could not force spawn start method: %s", exc)
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "The vllm backend requires the 'vllm' package, which is "
                "Linux-only (no macOS/arm64 wheels). On a Linux machine run "
                "`pip install -e \".[vllm]\"` and retry."
            ) from exc

        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

        self._model_id = model_id
        self._dtype_str = dtype
        self._device_requested = device
        self._use_chat_template = use_chat_template
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enforce_eager = enforce_eager
        self._max_model_len = max_model_len
        self._intervention = normalize_intervention(intervention)

        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "revision": revision,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "seed": seed,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        # vLLM >= 0.7 accepts device="auto"|"cuda"|"cpu". Older releases
        # do not; fall back without it rather than failing.
        pass_device = device if device != "auto" else None
        if pass_device is not None:
            llm_kwargs["device"] = pass_device
        try:
            self._llm = LLM(**llm_kwargs)
        except TypeError as exc:
            if pass_device is not None and "device" in str(exc):
                log.warning("vLLM release ignores device kwarg; retrying "
                            "without it (requested device=%s recorded as "
                            "metadata only)", device)
                del llm_kwargs["device"]
                self._llm = LLM(**llm_kwargs)
            else:
                raise

        self._resolved_revision = resolve_hub_revision(
            model_id, revision, kind="model")
        self._tokenizer_revision = (
            tokenizer_revision or revision or self._resolved_revision)
        log.info("vLLM engine started: %s rev=%s dtype=%s tp=%d",
                 model_id, self._resolved_revision, dtype,
                 tensor_parallel_size)

    # -- ModelAdapter contract -----------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return self._resolved_revision

    @property
    def tokenizer_revision(self) -> str | None:
        return self._tokenizer_revision

    @property
    def intervention_type(self) -> str:
        return str(self._intervention.get("type", "none"))

    def get_metadata(self) -> dict[str, Any]:
        base = super().get_metadata()
        base.update({
            "backend": "vllm",
            "vllm_version": _vllm_version(),
            "dtype": self._dtype_str,
            "device": self._resolve_device(),
            "device_requested": self._device_requested,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "enforce_eager": self._enforce_eager,
            "max_model_len": self._max_model_len,
            "use_chat_template": self._use_chat_template,
        })
        return base

    def _resolve_device(self) -> str:
        if self._device_requested != "auto":
            return self._device_requested
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "unknown"

    # -- generation ------------------------------------------------------
    def _sampling_params(self, config: GenerationConfig):
        from vllm import SamplingParams
        if config.do_sample:
            temperature, top_p = config.temperature, config.top_p
        else:
            temperature, top_p = 0.0, 1.0  # greedy
        return SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
        )

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        if not prompts:
            return []
        tokenizer = self._llm.get_tokenizer()
        formatted = [format_for_chat(tokenizer, p, self._use_chat_template)
                     for p in prompts]
        # Best-effort BOS guard (paper-v2 confound); engine owns batching
        # so we probe only the first prompt and never fail the run.
        try:
            check_bos_present(
                tokenizer, [tokenizer.encode(formatted[0])],
                context=f"model={self._model_id}")
        except Exception as exc:
            log.warning("BOS probe skipped (%s)", exc)
        params = self._sampling_params(config)
        # vLLM returns outputs in input order.
        t0 = time.perf_counter()
        request_outputs = self._llm.generate(formatted, params)
        elapsed = time.perf_counter() - t0
        results = []
        in_list: list[int] = []
        out_list: list[int] = []
        counts_ok = True
        for prompt, req in zip(formatted, request_outputs):
            if not req.outputs:
                raise RuntimeError(
                    f"vLLM returned no completions for prompt: {prompt[:120]!r}")
            results.append(req.outputs[0].text)
            plen = (len(req.prompt_token_ids)
                    if getattr(req, "prompt_token_ids", None) else None)
            comp_ids = getattr(req.outputs[0], "token_ids", None)
            clen = len(comp_ids) if comp_ids is not None else None
            if plen is None or clen is None:
                counts_ok = False
            else:
                in_list.append(plen)
                out_list.append(clen)
        # Engine prefill/decode are fused; only total time is measurable.
        self.last_batch_info = BatchTiming(
            input_tokens=in_list if counts_ok else None,
            output_tokens=out_list if counts_ok else None,
            prefill_seconds=None,
            generation_seconds=elapsed,
        )
        return results
