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
from eval_harness.prompting.chat import check_bos_present
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
    """Paper-exact alpha ramp (App. B.3): α_t = min(t/N, 1)·α.

    ``position`` is the 0-based absolute real position, so position 0
    yields factor 0 (warm-up: no mixing on the first step) and full
    strength is reached at ``position == ramp_tokens``. ``ramp_tokens
    <= 0`` disables ramping (factor 1.0 everywhere).
    """
    if ramp_tokens <= 0:
        return 1.0
    return min(position / ramp_tokens, 1.0)


def ramp_batch(positions: torch.Tensor, ramp_tokens: int) -> torch.Tensor:
    """Per-row ramp factors from absolute real positions (0-based)."""
    if ramp_tokens <= 0:
        return torch.ones_like(positions, dtype=torch.float32)
    return torch.clamp(
        positions.to(torch.float32) / float(ramp_tokens), max=1.0)


def mix_destination_batched(
    destination: torch.Tensor,
    source: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: float,
    normalization: str = "destination_l2",
    eps: float = EPS,
) -> torch.Tensor:
    """Row-wise ``alpha * f(source) + beta * destination`` ([B, H]).

    Norms are per-row (unlike the single-vector helper); ``alpha`` may
    be a per-row factor (ramping) or a scalar.
    """
    if normalization == "identity":
        scaled = source
    elif normalization == "destination_l2":
        src_f = source.to(torch.float32)
        dst_f = destination.to(torch.float32)
        src_norm = torch.linalg.vector_norm(src_f, dim=-1, keepdim=True)
        dst_norm = torch.linalg.vector_norm(dst_f, dim=-1, keepdim=True)
        scaled = (source * (dst_norm / (src_norm + eps))).to(source.dtype)
    else:
        raise ValueError(f"Unknown normalization {normalization!r}")
    if not torch.is_tensor(alpha):
        alpha = torch.full((destination.shape[0], 1), float(alpha),
                           dtype=destination.dtype, device=destination.device)
    else:
        alpha = alpha.to(dtype=destination.dtype,
                         device=destination.device).reshape(-1, 1)
    return alpha * scaled + beta * destination


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def _resolve_decoder_layers(model: Any, model_id: str) -> Any:
    """Locate the transformer block list across HF layout variants.

    Layouts differ by family: Llama-likes expose ``model.layers``;
    Gemma3's multimodal wrapper nests the text stack under
    ``model.language_model.layers``; GPT-2-likes use
    ``transformer.h``. Fail loudly listing what was tried.
    """
    candidates = (
        ("model.layers", lambda m: m.model.layers),
        ("model.language_model.layers",
         lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("transformer.h", lambda m: m.transformer.h),
    )
    for label, get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            log.info("decoder blocks for %s resolved via %s (%d blocks)",
                     model_id, label, len(layers))
            return layers
    raise RuntimeError(
        f"Recirculation requires a known decoder-block layout; "
        f"tried {[c[0] for c in candidates]} on {model_id}")


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
        decoder_layers = _resolve_decoder_layers(self.model, model_id)
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
    # All hooks operate on batches [B, 1, H]; padded/finished rows are
    # masked out via state["real"] so they neither mix nor pollute the
    # stored source. With no pads this reduces exactly to the
    # single-sequence path (verified by the batch-invariance tests).
    #
    # Two schedules share these hooks; only capture timing differs:
    # - cross_step: every forward mixes with the previously stored
    #   source (deep@t-1 into shallow@t); every forward stores anew.
    # - two_pass: pass 1 records only; pass 2 mixes with the same-step
    #   source captured moments earlier. Pass-2 lower layers recompute
    #   bit-identical states (deterministic rerun on identical inputs),
    #   so no boundary storage is needed.
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
        if self._recirc.type == "recirculation":
            if (self._recirc.schedule == "two_pass"
                    and st.get("pass", 1) == 1):
                return args, kwargs  # capture pass: record only
            if bool(st["has_source"].any()):
                cfg = self._recirc
                factors = ramp_batch(st["pos"], cfg.ramp_tokens)
                mixed = mix_destination_batched(
                    hidden[:, -1, :],
                    st["prev_source"].to(hidden.dtype),
                    cfg.alpha * factors,
                    cfg.effective_beta, cfg.normalization)
                gate = st["has_source"].to(hidden.dtype).reshape(-1, 1, 1)
                new_hidden = gate * mixed.unsqueeze(1) + (1.0 - gate) * hidden
                self._maybe_debug(st, hidden, new_hidden)
                hidden = new_hidden
        if key == "kwargs":
            return args, {**kwargs, "hidden_states": hidden}
        return (hidden,) + tuple(args[1:]), kwargs

    def _source_capture_hook(self, module, args, output):
        st = self._state
        if st is None or not st.get("active"):
            return
        if (self._recirc.schedule == "two_pass"
                and st.get("pass", 1) == 2):
            return  # source already stored by this step's pass 1
        hidden = output[0] if isinstance(output, tuple) else output
        try:
            real = st["real"]
            st["prev_source"][real] = hidden[real, -1, :].detach().to(
                st["prev_source"].dtype)
            st["has_source"][real] = True
        except (IndexError, RuntimeError) as exc:
            log.warning("source capture failed: %s", exc)

    def _maybe_debug(self, st, dst, mixed):
        if self._debug_steps <= 0 or len(self.last_debug) >= 1024:
            return
        if st.get("debug_count", 0) >= self._debug_steps:
            return
        with torch.no_grad():
            real_idx = torch.nonzero(st["real"]).flatten().tolist()
            for r in real_idx:
                if st.get("debug_count", 0) >= self._debug_steps:
                    break
                if len(self.last_debug) >= 1024:
                    break
                s = st["prev_source"][r].to(torch.float32)
                d = dst[r, -1, :].detach().to(torch.float32)
                m = mixed[r, -1, :].detach().to(torch.float32)
                dn, sn = d.norm().item(), s.norm().item()
                cos = (F.cosine_similarity(s, d, dim=0).item()
                       if dn > 0 and sn > 0 else 0.0)
                self.last_debug.append({
                    "batch_row": r,
                    "position": int(st["pos"][r].item()),
                    "alpha_effective": float(
                        self._recirc.alpha * ramp_factor(
                            int(st["pos"][r].item()),
                            self._recirc.ramp_tokens)),
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
        try:
            hidden_size: int | None = self._hidden_size()
        except Exception:
            hidden_size = None
        text_cfg = getattr(getattr(self.model, "config", None),
                           "text_config", None)
        context_length = getattr(text_cfg, "max_position_embeddings", None)
        if context_length is None:
            context_length = getattr(
                getattr(self.model, "config", None),
                "max_position_embeddings", None)
        schedule = self._recirc.schedule
        base.update({
            "hidden_size": hidden_size,
            "num_layers": self._num_layers,
            "context_length": context_length,
            "layer_indexing_convention": "block_0based",
            "recurrence_variant": f"{schedule}_tokenwise_serial",
        })
        return base

    @staticmethod
    def _truncate_cache(cache: Any, length: int) -> Any:
        """Drop cached positions >= length in place (two-pass step redo).

        Prefers the cache's native ``crop()`` (transformers >= 4.45
        layout with per-layer caches); falls back to slicing legacy
        ``key_cache``/``value_cache`` lists. Anything else raises loudly
        rather than silently corrupting generation state.
        """
        crop = getattr(cache, "crop", None)
        if callable(crop):
            try:
                crop(length)
                return cache
            except Exception as exc:
                raise RuntimeError(
                    f"two-pass cache crop({length}) failed on "
                    f"{type(cache).__name__}: {exc}") from exc
        for attr in ("key_cache", "value_cache"):
            stacks = getattr(cache, attr, None)
            if (isinstance(stacks, list) and stacks
                    and hasattr(stacks[0], "shape")):
                for i, t in enumerate(stacks):
                    stacks[i] = t[:, :, :length, :]
            else:
                raise TypeError(
                    "two-pass recirculation needs a croppable KV cache "
                    f"(cache lacks usable {attr!r}: "
                    f"{type(cache).__name__})")
        return cache

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
        formatted = [self._format_prompt(p) for p in prompts]
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True)
        check_bos_present(
            self.tokenizer, enc["input_ids"],
            context=f"model={self._model_id}",
            mask=enc.get("attention_mask"))
        texts, in_list, out_list, pre_s, dec_s = self._generate_batch(
            enc, config, sampler)
        # Unlike the HF adapter's fused generate(), this loop genuinely
        # separates serial prefill from decode (P2 §31).
        self.last_batch_info = BatchTiming(
            input_tokens=in_list,
            output_tokens=out_list,
            prefill_seconds=pre_s,
            generation_seconds=dec_s,
        )
        return texts

    def _generate_batch(self, enc: Any, config: GenerationConfig,
                        sampler: torch.Generator,
                        ) -> tuple[list[str], list[int], list[int],
                                   float, float]:
        """Batched serial loop. Returns (texts, in, out, pre_s, dec_s).

        Left padding aligns rows; pads never mix, never update stored
        source, and never enter attention (mask). Finished decode rows
        feed pad and are ignored. Row order is preserved throughout.
        """
        input_ids = enc["input_ids"]
        attn0 = enc.get("attention_mask")
        if attn0 is None:
            attn0 = torch.ones_like(input_ids)
        B, L = input_ids.shape
        lengths = attn0.sum(dim=1).tolist()
        if min(lengths) == 0:
            raise ValueError("recirculation needs ≥1 real token per prompt")
        device = self._device
        H = self._hidden_size()
        model_dtype = next(self.model.parameters()).dtype
        # Flags live on the model device (hooks run device-side); the
        # driver loop converts to Python scalars/lists when branching.
        self._state = {
            "active": True,
            "prev_source": torch.zeros(B, H, dtype=model_dtype,
                                       device=device),
            "has_source": torch.zeros(B, dtype=torch.bool, device=device),
            "real": torch.ones(B, dtype=torch.bool, device=device),
            "pos": torch.zeros(B, dtype=torch.long, device=device),
            "debug_count": 0,
            "pass": 1,
        }
        two_pass = (self._recirc.type == "recirculation"
                    and self._recirc.schedule == "two_pass")
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        new_ids: list[list[int]] = [[] for _ in range(B)]
        past = None
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        try:
            # Serial prefill over the padded block.
            t_pre = time.perf_counter()
            cum_mask = attn0[:, :0]
            logits = None
            for t in range(L):
                real = attn0[:, t].to(torch.bool).to(device)
                cum_mask = torch.cat([cum_mask, attn0[:, t:t + 1]], dim=1)
                self._state["real"] = real
                logits, past = self._step_with_passes(
                    input_ids[:, t:t + 1].to(device),
                    cum_mask.to(device), t, past, two_pass)
                self._state["pos"][real] += 1
            prefill_s = time.perf_counter() - t_pre
            # Decode until every row hits EOS or the cap.
            t_dec = time.perf_counter()
            cur = torch.full((B, 1), pad_id, dtype=torch.long)
            for i in range(config.max_new_tokens):
                assert logits is not None
                nxt = self._select_rows(logits, finished, config, sampler)
                fin_now = finished.tolist()
                for r in range(B):
                    if not fin_now[r]:
                        new_ids[r].append(nxt[r])
                finished |= torch.tensor(
                    [n == self.tokenizer.eos_token_id for n in nxt],
                    device=device)
                if bool(finished.all()):
                    break
                for r in range(B):
                    cur[r, 0] = nxt[r] if not finished[r] else pad_id
                real = ~finished
                cum_mask = torch.cat(
                    [cum_mask, torch.ones(B, 1)], dim=1)
                self._state["real"] = real
                logits, past = self._step_with_passes(
                    cur.to(device), cum_mask.to(device), L + i, past,
                    two_pass)
                self._state["pos"][real] += 1
            decode_s = time.perf_counter() - t_dec
        finally:
            # Release the KV cache; hooks tolerate _state = None.
            self._state = None
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True)
                 for ids in new_ids]
        return (texts, [int(v) for v in lengths],
                [len(ids) for ids in new_ids], prefill_s, decode_s)

    def _hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        for obj in (cfg, getattr(cfg, "text_config", None)):
            hs = getattr(obj, "hidden_size", None)
            if hs is not None:
                return int(hs)
        for obj in (getattr(getattr(self.model, "model", None),
                            "embed_tokens", None),
                    getattr(getattr(getattr(self.model, "model", None),
                                    "language_model", None),
                            "embed_tokens", None)):
            if obj is not None:
                return int(obj.weight.shape[1])
        raise RuntimeError(
            f"cannot determine hidden size for {self._model_id}")

    def _step_with_passes(self, tokens: torch.Tensor, mask: torch.Tensor,
                            position: int, past: Any,
                            two_pass: bool) -> tuple[torch.Tensor, Any]:
        """One input step: capture pass, then optional mix rerun.

        Pass 1 always runs the full stack (records source, warms cache).
        In two-pass mode at positions > 0, the cache is truncated back to
        `position` and the same token is rerun with the same-step source
        mixed at the boundary; the rerun overwrites the upper-layer KV
        and supplies the logits. Position 0 is a warm-up (pass 1 only).
        """
        assert self._state is not None
        st = self._state
        st["pass"] = 1
        logits = self._forward(tokens, mask, position, past)
        past = st["past"]
        if two_pass and position > 0:
            self._truncate_cache(past, position)
            st["pass"] = 2
            logits = self._forward(tokens, mask, position, past)
            past = st["past"]
            st["pass"] = 1
        return logits, past

    def _forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                 position: int, past: Any) -> torch.Tensor:
        assert self._state is not None
        out = self.model(
            input_ids=tokens,
            attention_mask=mask,
            cache_position=torch.tensor([position], device=self._device),
            past_key_values=past,
            use_cache=True,
        )
        self._state["past"] = out.past_key_values
        return out.logits[:, -1, :].detach()

    def _select_rows(self, logits: torch.Tensor, finished: torch.Tensor,
                     config: GenerationConfig,
                     sampler: torch.Generator) -> list[int]:
        if not config.do_sample or config.temperature <= 0.0:
            return torch.argmax(logits, dim=-1).tolist()
        # Sample on CPU in fixed row order: deterministic regardless of
        # batching. Finished rows still draw (ignored) to keep the
        # sampler stream position-independent of finish order.
        probs = torch.softmax(
            logits.float() / max(config.temperature, 1e-9), dim=-1).cpu()
        if config.top_p < 1.0:
            order = torch.argsort(probs, descending=True)
            ranked = probs.gather(1, order)
            keep = torch.cumsum(ranked, dim=-1) <= config.top_p
            keep[:, 0] = True
            denom = (ranked * keep).sum(dim=-1, keepdim=True)
            ranked = ranked * keep / denom
            probs = torch.zeros_like(ranked).scatter_(1, order, ranked)
        return [int(torch.multinomial(probs[r], 1,
                                      generator=sampler).item())
                for r in range(probs.shape[0])]
