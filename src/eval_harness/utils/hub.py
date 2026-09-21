"""Hugging Face Hub helpers (shared by model adapters)."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_hub_revision(
    repo_id: str,
    pinned: str | None,
    *,
    kind: str = "model",
) -> str | None:
    """Return the pinned revision, else the Hub's current commit sha.

    Never raises: returns None when offline or the lookup fails.
    Never overwrites a user-specified revision.
    """
    if pinned:
        return pinned
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id) if kind == "model" \
            else api.dataset_info(repo_id)
        return info.sha
    except Exception as exc:
        log.warning("could not resolve %s revision for %s: %s",
                    kind, repo_id, exc)
        return None
