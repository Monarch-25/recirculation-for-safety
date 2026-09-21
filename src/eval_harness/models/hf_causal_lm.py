"""Hugging Face causal-LM adapter.

Owns generation only: tokenizer/model loading, device/dtype
resolution, batched greedy (or sampled) decoding. No benchmark logic.

Device resolution:
    - ``device="auto"``: CUDA if available, else MPS if available,
      else CPU.
    - explicit ``"cuda"``/``"mps"``/``"cpu"``: use it if available,
      else raise (never silently override explicit user config,
      except auto mode which degrades gracefully).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness.core.interfaces import GenerationConfig, ModelAdapter
from eval_harness.prompting.chat import format_for_chat
from eval_harness.runtime import reproducibility
from eval_harness.utils.hub import resolve_hub_revision

log = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str) -> str:
    """Resolve a device string without silently overriding explicit config."""
    requested = (requested or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()
    if requested == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if requested == "mps" and not mps_ok:
        raise RuntimeError("device='mps' requested but MPS is not available")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device '{requested}'")
    return requested


def resolve_revision(model_id: str, pinned: str | None) -> str | None:
    """Return pinned revision, else resolved Hub commit sha if reachable."""
    return resolve_hub_revision(model_id, pinned, kind="model")


class HFCausalLMAdapter(ModelAdapter):
    """Baseline HF causal-LM adapter (intervention_type='none')."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        dtype: str = "float32",
        device: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        attn_implementation: str | None = None,
        intervention: dict[str, Any] | None = None,
    ) -> None:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype}'")
        self._model_id = model_id
        self._pinned_revision = revision
        self._tokenizer_revision = tokenizer_revision or revision
        self._dtype_str = dtype
        self._device = resolve_device(device)
        self._trust_remote_code = trust_remote_code
        self._use_chat_template = use_chat_template
        self._attn_implementation = attn_implementation
        self._intervention = dict(intervention or {"type": "none"})

        torch_dtype = _DTYPE_MAP[dtype]
        log.info("loading tokenizer %s rev=%s", model_id, self._tokenizer_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=self._tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models require left padding for correct batched
        # generation (right padding corrupts positional alignment).
        self.tokenizer.padding_side = "left"
        log.info("loading model %s rev=%s dtype=%s device=%s",
                 model_id, revision, dtype, self._device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **({"attn_implementation": attn_implementation}
               if attn_implementation else {}),
        )
        self.model.to(self._device)
        self.model.eval()

        self._resolved_revision = resolve_revision(model_id, revision)
        if self._tokenizer_revision is None:
            # Same repo → same commit unless pinned otherwise.
            self._tokenizer_revision = self._resolved_revision
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("model loaded: %s params=%d device=%s",
                 model_id, n_params, self._device)
        self._num_params = n_params

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
            "dtype": self._dtype_str,
            "device": self._device,
            "num_parameters": self._num_params,
            "use_chat_template": self._use_chat_template,
            "attn_implementation": self._attn_implementation,
        })
        return base

    @property
    def num_parameters(self) -> int:
        return self._num_params

    # -- generation ------------------------------------------------------
    def _format_prompt(self, prompt: str) -> str:
        return format_for_chat(
            self.tokenizer, prompt, self._use_chat_template)

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        if not prompts:
            return []
        # Deterministic seeding per call.
        random.seed(config.seed)
        np.random.seed(config.seed % (2 ** 32))
        torch.manual_seed(config.seed)
        reproducibility.seed_everything(config.seed)

        formatted = [self._format_prompt(p) for p in prompts]
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
        output_ids = self.model.generate(
            input_ids, attention_mask=attention_mask, **gen_kwargs)
        # generate() returns [padded input (width = input_width) + new
        # tokens]; slicing at input_width yields only new tokens for both
        # left- and right-padded batches.
        results: list[str] = []
        input_width = input_ids.shape[1]
        for row in output_ids:
            new_tokens = row[input_width:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)
        return results
