"""Fixed training-free deep-to-shallow Recirculation adapter.

Reuses :class:`HFCausalLMAdapter` loading/device/tokenizer machinery and
overrides only generation with a serial recurrent loop. Three schedules:

``cross_step`` (delayed cross-token)::

    step t:     full-stack forward, capture deep source residual h_s(t)
    step t + 1: mix stored h_s(t) into the shallow destination boundary,
                rerun blocks d+1..N from the mixed state

    * One forward pass per input step: serial prefill AND serial decode.
    * Position 0 is a warm-up (no stored source exists yet).
    * NOTE: the ModelCloud/Recirculation reproduction explicitly withdrew
      this "delayed cross-token intervention" as recirculation evidence —
      it is NOT the paper's method.

``two_pass`` (same-token, second-iteration readout)::

    pass 1: full stack, capture deep source h_s(t) AND readout
    pass 2: mix the same-step source into the destination boundary,
            rerun blocks d+1..N, overwrite KV, readout from the RERUN

    * Readout comes from the mixed (second) iteration — also NOT the
      paper's readout policy.

``mozer`` (paper-exact, Mozer et al. eq. 1-2)::

    pass 1: full stack, capture h_s(t) and h_d(t), readout (FIRST pass)
    pass 2: mix (alpha*f(h_s) + beta*h_d) at the boundary, replay
            blocks d+1..N, overwrite KV; NO readout from the replay

    * "The read out occurs following the first iteration of a stack"
      (Fig. 3): the additional iteration only replaces the token's
      upper-layer KV, which subsequent tokens attend to.
    * Default convex mixtures couple beta_t = 1 - alpha_t under ramping
      (alpha = alpha_max * min(pos/ramp_tokens, 1)).

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


def apply_stop_strings(text: str, stops: Sequence[str]) -> str:
    """Truncate ``text`` before the earliest stop string (Evalution parity).

    No-op when no stop string occurs. Operates on the jointly-decoded
    row text so the cut is exact even when a stop spans token pieces.
    """
    cut = len(text)
    for s in stops:
        if s:
            j = text.find(s)
            if j != -1 and j < cut:
                cut = j
    return text[:cut]


def mix_destination_batched(
    destination: torch.Tensor,
    source: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
    normalization: str = "destination_l2",
    eps: float = EPS,
) -> torch.Tensor:
    """Row-wise ``alpha * f(source) + beta * destination`` ([B, H]).

    Norms are per-row (unlike the single-vector helper); ``alpha``/``beta``
    may be per-row factors (ramping / convex coupling) or scalars.
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
    if not torch.is_tensor(beta):
        beta = float(beta)
    else:
        beta = beta.to(dtype=destination.dtype,
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
    # Capture timing per schedule (the replay/rerun passes are identical):
    # - cross_step: every forward mixes with the previously stored
    #   source (deep@t-1 into shallow@t); every forward stores anew.
    # - two_pass: pass 1 records only; pass 2 mixes with the same-step
    #   source captured moments earlier and supplies the logits.
    # - mozer:   pass 1 records only; pass 2 mixes with the same-step
    #   source (KV overwritten) but the readout stays on pass 1, exactly
    #   as in Figure 3 of the paper.
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
            if (self._recirc.schedule in ("two_pass", "mozer")
                    and st.get("pass", 1) == 1):
                return args, kwargs  # capture pass: record only
            if bool(st["has_source"].any()):
                cfg = self._recirc
                factors = ramp_batch(st["pos"], cfg.ramp_tokens)
                alpha_t = cfg.alpha * factors
                # Paper-default convex mixture couples beta_t = 1 - alpha_t
                # while alpha ramps (repo: beta_t = 1 - alpha_t unless an
                # explicit non-default beta is set). Non-convex stays fixed.
                if (cfg.mixture == "convex" and cfg.beta is None):
                    beta_t = 1.0 - alpha_t
                else:
                    beta_t = cfg.effective_beta
                mixed = mix_destination_batched(
                    hidden[:, -1, :],
                    st["prev_source"].to(hidden.dtype),
                    alpha_t, beta_t, cfg.normalization)
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
        if (self._recirc.schedule in ("two_pass", "mozer")
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
    def _arm_past_recording(cache: Any) -> None:
        """Best-effort ``activate_past_recording`` before a redoable forward.

        Gemma3 sliding-window layers discard past states on every update
        unless recording is armed; without this, the post-forward ``crop``
        raises once the window is full. No-op for ``None`` caches and for
        cache classes without the method (older transformers).
        """
        if cache is None:
            return
        activ = getattr(cache, "activate_past_recording", None)
        if callable(activ):
            try:
                activ()
            except Exception:
                pass

    @staticmethod
    def _restrict_cache(cache: Any) -> None:
        """Best-effort ``crop(0)``: re-restrict sliding layers to the window.

        Must run after the replay (pass-2) forward while recording is still
        armed: sliding layers otherwise retain full history and the next
        forward's attention sees ``window + 1`` keys (513 vs 512 SDPA
        mismatch). ``crop(0)`` removes no tokens on full-attention layers
        (no-op) and drops already-superseded states on sliding layers.
        """
        if cache is None:
            return
        crop = getattr(cache, "crop", None)
        if callable(crop):
            try:
                crop(0)
            except Exception:
                pass

    @staticmethod
    def _truncate_cache(cache: Any, length: int) -> Any:
        """Remove the just-added token in place (two-pass step redo).

        Prefers native ``crop(-1)`` (remove-last), which is correct for
        full-attention layers on all supported transformers versions and
        additionally re-restricts Gemma3 sliding layers back toward the
        window. Requires recording to be armed (see ``_arm_past_recording``)
        once the sliding window is full; falls back to legacy absolute
        ``crop(length)`` and then to manual ``key_cache``/``value_cache``
        slicing. Raises loudly rather than silently corrupting state.
        """
        crop = getattr(cache, "crop", None)
        if callable(crop):
            # Preferred: remove exactly the last-added token (works for
            # full + sliding layers on both old and new transformers,
            # both of which accept negative crop).
            try:
                crop(-1)
                return cache
            except Exception:
                pass
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
        prompts: Sequence[str | list[dict[str, str]]],
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
        rerun = (self._recirc.type == "recirculation"
                    and self._recirc.schedule in ("two_pass", "mozer"))
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        new_ids: list[list[int]] = [[] for _ in range(B)]
        past = None
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        # Model EOS set (Gemma3: [1, 106]; tokenizer scalar alone misses
        # <end_of_turn> and generations never stop — see eos_token_ids()).
        eos_ids = set(self.eos_token_ids())
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
                    cum_mask.to(device), t, past, rerun)
                self._state["pos"][real] += 1
            prefill_s = time.perf_counter() - t_pre
            # Decode until every row hits EOS, a stop string, or the cap.
            t_dec = time.perf_counter()
            cur = torch.full((B, 1), pad_id, dtype=torch.long)
            stops = list(config.stop_strings or ())
            # Incremental per-row text for stop detection (trigger only;
            # final text is re-decoded jointly and cut exactly).
            row_tails: list[str] = ["" for _ in range(B)]
            for i in range(config.max_new_tokens):
                assert logits is not None
                nxt = self._select_rows(logits, finished, config, sampler)
                fin_now = finished.tolist()
                for r in range(B):
                    if not fin_now[r]:
                        new_ids[r].append(nxt[r])
                if stops:
                    for r in range(B):
                        if fin_now[r]:
                            continue
                        piece = self.tokenizer.decode(
                            [nxt[r]], skip_special_tokens=True)
                        row_tails[r] += piece
                        # Search a bounded tail (stops are short); exact
                        # cut happens on the jointly-decoded text below.
                        if any(s and s in row_tails[r][-256:] for s in stops):
                            fin_now[r] = True
                    finished = torch.tensor(fin_now, device=device)
                finished |= torch.tensor(
                    [n in eos_ids for n in nxt],
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
                    rerun)
                self._state["pos"][real] += 1
            decode_s = time.perf_counter() - t_dec
        finally:
            # Release the KV cache; hooks tolerate _state = None.
            self._state = None
        texts = []
        for ids in new_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(apply_stop_strings(text, stops))
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
                            rerun: bool) -> tuple[torch.Tensor, Any]:
        """One input step: capture pass, then optional mix rerun.

        Pass 1 always runs the full stack (records source, warms cache).
        In rerun modes (`two_pass`/`mozer`) at positions > 0, the cache is
        truncated back to `position` and the same token is rerun with the
        same-step source mixed at the boundary; the rerun overwrites the
        upper-layer KV. Position 0 is a warm-up (pass 1 only).

        Readout differs by schedule (paper Figure 3):
        - two_pass: the mixed (pass-2) rerun supplies the logits.
        - mozer:    the FIRST-pass logits supply the readout; the rerun only
                    replaces the token's upper KV, exactly as the paper
                    specifies ("the read out occurs following the first
                    iteration of a stack").
        """
        assert self._state is not None
        st = self._state
        st["pass"] = 1
        # Arm sliding-window rollback BEFORE the forward whose state a
        # later crop(-1) must restore; arming after the forward is too
        # late (states already discarded) and arming without a closing
        # crop(0) lets the cache grow past the window (513-vs-512 SDPA).
        if rerun and position > 0:
            self._arm_past_recording(past)
        logits = self._forward(tokens, mask, position, past)
        past = st["past"]
        if rerun and position > 0:
            self._truncate_cache(past, position)
            st["pass"] = 2
            rerun_logits = self._forward(tokens, mask, position, past)
            # Re-restrict sliding layers to the window; full layers no-op.
            self._restrict_cache(st["past"])
            past = st["past"]
            st["pass"] = 1
            if self._recirc.schedule == "two_pass":
                logits = rerun_logits
        return logits, past

    def _forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                 position: int, past: Any) -> torch.Tensor:
        assert self._state is not None
        # Per-row absolute positions. Batches are left-padded, so rows sit
        # at different real positions at the same block step; the shared
        # scalar cache_position (correct as the uniform cache-slot index)
        # would misplace RoPE for every row except the longest. Derive
        # each row's real position from the cumulative mask instead.
        real_pos = (mask.sum(dim=1, keepdim=True) - 1).clamp_min(0).to(
            torch.long)
        out = self.model(
            input_ids=tokens,
            attention_mask=mask,
            cache_position=torch.tensor([position], device=self._device),
            position_ids=real_pos.to(self._device),
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
