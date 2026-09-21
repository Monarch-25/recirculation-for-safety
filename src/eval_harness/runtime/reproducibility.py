"""Reproducibility helpers: seeding, run ids, git metadata."""

from __future__ import annotations

import datetime
import random
import secrets
import subprocess
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def new_run_id(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(3)}"


def get_git_metadata(repo: str | Path | None = None) -> dict[str, Any]:
    """Return git commit/branch/dirty; never raises (fields None on failure)."""
    meta: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        def _run(*args: str) -> str | None:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10,
                cwd=str(repo) if repo else None,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip() or None
        meta["commit"] = _run("rev-parse", "HEAD")
        meta["branch"] = _run("rev-parse", "--abbrev-ref", "HEAD")
        status = _run("status", "--porcelain")
        meta["dirty"] = (len(status) > 0) if status is not None else None
        if status is None and meta["commit"] is None:
            meta["dirty"] = None
        elif status is None:
            meta["dirty"] = None
        elif status == "":
            meta["dirty"] = False
        else:
            meta["dirty"] = True
    except Exception:
        pass
    return meta
