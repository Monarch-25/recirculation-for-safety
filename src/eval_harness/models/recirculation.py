"""Fixed training-free deep-to-shallow Recirculation adapter.

Reuses :class:`HFCausalLMAdapter` loading/device/tokenizer machinery and
overrides only generation with a serial cross-step recurrence:

.. code-block:: text

    step t:     full-stack forward, capture deep source residual h_s(t)
    step t + 1: mix stored h_s(t) into the shallow destination boundary,
                rerun blocks d+1..N from the mixed state

Interpretation (honest labeling, plan P2 §26):

* One forward pass per input step: serial prefill AND serial decode
  (matches the paper's "serial processing of the prefill context").
* Position 0 is a warm-up (no stored source exists yet).
* Cross-step propagation happens two ways: the mixed residual at t+1,
  and the KV cache, which stores post-mixing upper-layer states that
  later positions attend to.
* This is deliberately NOT same-step mixing (deep@t -> shallow@t in one
  pass) and NOT depth looping. If a future reading of the paper demands
  the two-stack variant, this file documents where it would diverge.

Layer convention: 0-based transformer blocks (``model.model.layers``).
Destination boundary = output of block ``d`` = input to block ``d+1``
(intercepted via forward pre-hook). Source = output of block ``s``
(captured via forward hook). ``source > destination`` is enforced at
config load; bounds are asserted here at init.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from eval_harness.core.config import (
    InterventionConfig,
    normalize_intervention,
)
from eval_harness.core.interfaces import BatchTiming, GenerationConfig
from eval_harness.models.hf_causal_lm import HFCausalLMAdapter
from eval_harness.runtime import reproducibility

log = logging.getLogger(__name__)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Pure mixing math (unit-testable without a model)
# ---------------------------------------------------------------------------

def rescale_to_destination(
    source: torch.Tensor,
    destination: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Rescale source to the destination L2 norm (paper Eq. 2)."""
    src_norm = torch.linalg.vector_norm(source.to(torch.float32))
    dst_norm = torch.linalg.vector_norm(destination.to(torch.float32))
    scale = dst_norm / (src_norm + eps)
    return (source * scale).to(source.dtype)


def mix_destination(
    destination: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    beta: float,
    normalization: str = "destination_l2",
) -> torch.Tensor:
    """Return ``alpha * f(source) + beta * destination``."""
    if normalization == "identity":
        scaled = source
    elif normalization == "destination_l2":
        scaled = rescale_to_destination(source, destination)
    else:
        raise ValueError(f"Unknown normalization {normalization!r}")
    return alpha * scaled + beta * destination


def ramp_factor(position: int, ramp_tokens: int) -> float:
    """Linear alpha ramp over early positions (provisional schedule).

    ``ramp_tokens <= 0`` disables ramping (factor 1.0). Otherwise the
    factor grows 0 -> 1 over the first ``ramp_tokens`` positions.
    """
    if ramp_tokens <= 0:
        return 1.0
    return min(1.0, (position + 1) / ramp_tokens)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class RecirculationModelAdapter(HFCausalLMAdapter):
    """Frozen-weight Recirculation adapter (intervention='recirculation').

    ``intervention.type == 'none'`` is accepted and behaves as the
    baseline serial loop (no mixing ever) — the P2 §25 equivalence
    condition up to serial-vs-parallel prefill numerics.
    """

    def __init__(
        self,
        model_id: str,
        *,
        debug_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        # Fail fast on out-of-bounds layers using config metadata only
        # (no weight download); the post-load check below remains as a
        # backstop in case this probe cannot reach the Hub.
        _pre = InterventionConfig.from_dict(
            normalize_intervention(kwargs.get("intervention")))
        if _pre.type == "recirculation":
            self._assert_layer_bounds(
                model_id, _pre, kwargs.get("revision"),
                kwargs.get("trust_remote_code", False))
        super().__init__(model_id, **kwargs)
        self._recirc = InterventionConfig.from_dict(self._intervention)
        try:
            decoder_layers = self.model.model.layers
        except AttributeError as exc:
            raise RuntimeError(
                f"Recirculation requires model.model.layers; "
                f"{model_id} has an unsupported layout") from exc
        self._decoder_layers = decoder_layers
        self._num_layers = len(decoder_layers)
        if self._recirc.type == "recirculation":
            self._assert_layer_bounds(
                model_id, self._recirc, self._pinned_revision,
                self._trust_remote_code, known_layers=self._num_layers)
        self._debug_steps = max(0, int(debug_steps))
        self.last_debug: list[dict[str, Any]] = []
        # Per-generate-call mutable state (single-threaded use only).
        self._state: dict[str, Any] | None = None
        self._pre_handle = self._decoder_layers[
            (self._recirc.destination_layer or 0) + 1
        ].register_forward_pre_hook(self._destination_pre_hook,
                                    with_kwargs=True)
        src_idx = self._recirc.source_layer or 0
        self._cap_handle = self._decoder_layers[
            src_idx
        ].register_forward_hook(self._source_capture_hook)
        log.info("recirculation ready: %s layers=%d hidden=%s cfg=%s",
                 model_id, self._num_layers,
                 getattr(self.model.config, "hidden_size", "?"),
                 self._recirc.to_dict())

    @staticmethod
    def _assert_layer_bounds(
        model_id: str,
        cfg: InterventionConfig,
        revision: str | None,
        trust_remote_code: bool,
        known_layers: int | None = None,
    ) -> None:
        """Reject layer indices outside [0, num_blocks)."""
        if known_layers is None:
            try:
                from transformers import AutoConfig
                hub_cfg = AutoConfig.from_pretrained(
                    model_id, revision=revision,
                    trust_remote_code=trust_remote_code)
                known_layers = int(hub_cfg.num_hidden_layers)
            except Exception as exc:
                log.warning("layer-bound probe failed for %s (%s); "
                            "deferring to post-load check", model_id, exc)
                return
        for label, idx in (("source_layer", cfg.source_layer),
                           ("destination_layer", cfg.destination_layer)):
            if idx is None or not 0 <= idx < known_layers:
                raise ValueError(
                    f"intervention.{label}={idx} out of bounds for "
                    f"{model_id} with {known_layers} blocks")

    # -- hooks ----------------------------------------------------------
    def _destination_pre_hook(self, module, args, kwargs):
        st = self._state
        if st is None or not st.get("active"):
            return args, kwargs
        if "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
            key = "kwargs"
        else:
            hidden = args[0]
            key = "args"
        if self._recirc.type == "recirculation" and st["prev_source"] is not None:
            cfg = self._recirc
            alpha = cfg.alpha * ramp_factor(st["position"], cfg.ramp_tokens)
            mixed = mix_destination(
                hidden, st["prev_source"].to(hidden.dtype),
                alpha, cfg.effective_beta, cfg.normalization)
            self._maybe_debug(st, hidden, mixed, alpha)
            hidden = mixed
        if key == "kwargs":
            return args, {**kwargs, "hidden_states": hidden}
        return (hidden,) + tuple(args[1:]), kwargs

    def _source_capture_hook(self, module, args, output):
        st = self._state
        if st is None or not st.get("active"):
            return
        hidden = output[0] if isinstance(output, tuple) else output
        try:
            st["prev_source"] = hidden[0, -1, :].detach().clone()
        except (IndexError, RuntimeError) as exc:
            log.warning("source capture failed: %s", exc)

    def _maybe_debug(self, st, dst, mixed, alpha):
        if self._debug_steps <= 0 or len(self.last_debug) >= 1024:
            return
        if st.get("debug_count", 0) >= self._debug_steps:
            return
        with torch.no_grad():
            src = st["prev_source"].to(torch.float32)
            d = dst.detach().to(torch.float32).reshape(-1)
            m = mixed.detach().to(torch.float32).reshape(-1)
            s = src.reshape(-1)
            dn, sn = d.norm().item(), s.norm().item()
            cos = (F.cosine_similarity(s, d, dim=0).item()
                   if dn > 0 and sn > 0 else 0.0)
            self.last_debug.append({
                "position": st["position"],
                "alpha_effective": alpha,
                "source_norm": sn,
                "destination_norm": dn,
                "scaled_source_norm": dn,
                "mixed_norm": m.norm().item(),
                "cosine": cos,
            })
            st["debug_count"] = st.get("debug_count", 0) + 1

    # -- metadata ---------------------------------------------------------
    def get_metadata(self) -> dict[str, Any]:
        base = super().get_metadata()
        model_cfg = getattr(self.model, "config", None)
        base.update({
            "hidden_size": getattr(model_cfg, "hidden_size", None),
            "num_layers": self._num_layers,
            "context_length": getattr(
                model_cfg, "max_position_embeddings", None),
            "layer_indexing_convention": "block_0based",
            "recurrence_variant": "cross_step_tokenwise_serial",
        })
        return base

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[str]:
        if not prompts:
            return []
        reproducibility.seed_everything(config.seed)
        sampler = torch.Generator().manual_seed(config.seed)
        self.last_debug = []
        texts: list[str] = []
        in_list: list[int] = []
        out_list: list[int] = []
        prefill_sum, decode_sum = 0.0, 0.0
        try:
            for p in prompts:
                text, n_in, n_out, t_pre, t_dec = self._generate_one(
                    p, config, sampler)
                texts.append(text)
                in_list.append(n_in)
                out_list.append(n_out)
                prefill_sum += t_pre
                decode_sum += t_dec
        finally:
            self._state = None
        # Unlike the HF adapter's fused generate(), this loop genuinely
        # separates serial prefill from decode (P2 §31).
        self.last_batch_info = BatchTiming(
            input_tokens=in_list,
            output_tokens=out_list,
            prefill_seconds=prefill_sum,
            generation_seconds=decode_sum,
        )
        return texts

    def _generate_one(
        self,
        prompt: str,
        config: GenerationConfig,
        sampler: torch.Generator,
    ) -> tuple[str, int, int, float, float]:
        """Generate one continuation; also return (in, out, pre, dec)."""
        formatted = self._format_prompt(prompt)
        ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            return "", 0, 0, 0.0, 0.0
        from eval_harness.prompting.chat import check_bos_present
        check_bos_present(
            self.tokenizer, [ids.tolist()], context=f"model={self._model_id}")
        self._state = {"active": True, "prev_source": None,
                       "position": 0, "debug_count": 0}
        past = None
        new_ids: list[int] = []
        try:
            # Serial prefill: every prompt token flows through the loop,
            # so recirculation state builds exactly as in decode. The
            # logits from the final prompt position predict token one.
            logits = None
            t_pre = time.perf_counter()
            for pos in range(ids.numel()):
                logits = self._step(ids[pos:pos + 1], past, pos)
                past = self._state["past"]
            prefill_s = time.perf_counter() - t_pre
            # Decode: sample from the previous position's logits, then
            # advance exactly one position per generated token.
            t_dec = time.perf_counter()
            for i in range(config.max_new_tokens):
                assert logits is not None
                nxt = self._select(logits, config, sampler)
                new_ids.append(nxt)
                if nxt == (self.tokenizer.eos_token_id):
                    break
                logits = self._step(
                    torch.tensor([nxt]), past, ids.numel() + i)
                past = self._state["past"]
            decode_s = time.perf_counter() - t_dec
        finally:
            self._state["active"] = False
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, ids.numel(), len(new_ids), prefill_s, decode_s

    def _step(self, token: torch.Tensor, past: Any, position: int) -> torch.Tensor:
        assert self._state is not None
        self._state["position"] = position
        seq_len = (past.get_seq_length() + 1) if past is not None else 1
        attention_mask = torch.ones(1, seq_len, device=self._device)
        cache_position = torch.tensor([seq_len - 1], device=self._device)
        out = self.model(
            input_ids=token.reshape(1, 1).to(self._device),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past,
            use_cache=True,
        )
        self._state["past"] = out.past_key_values
        return out.logits[0, -1, :].detach()

    @staticmethod
    def _select(logits: torch.Tensor, config: GenerationConfig,
                sampler: torch.Generator) -> int:
        if not config.do_sample or config.temperature <= 0.0:
            return int(torch.argmax(logits).item())
        # Sample on CPU: tiny tensor, avoids device/generator mismatches.
        probs = torch.softmax(
            logits.float() / max(config.temperature, 1e-9), dim=-1).cpu()
        if config.top_p < 1.0:
            order = torch.argsort(probs, descending=True)
            ranked = probs[order]
            keep = torch.cumsum(ranked, dim=-1) <= config.top_p
            keep[0] = True
            ranked = ranked * keep / (ranked * keep).sum()
            probs = torch.zeros_like(probs).scatter_(0, order, ranked)
        return int(torch.multinomial(probs, 1, generator=sampler).item())
