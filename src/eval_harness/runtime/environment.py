"""Runtime environment metadata collection (single place for probing)."""

from __future__ import annotations

import os
import platform
import socket
import sys
import time
from importlib import metadata
from typing import Any

from eval_harness.runtime.hardware import get_hardware_info


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    hw = get_hardware_info()
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "datasets_version": _pkg_version("datasets"),
        "tokenizers_version": _pkg_version("tokenizers"),
        "accelerate_version": _pkg_version("accelerate"),
        "pydantic_version": _pkg_version("pydantic"),
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "hostname": socket.gethostname(),
        "cpu": hw.get("cpu") or os.cpu_count(),
        "ram_gb": _ram_gb(),
        "cuda_available": hw.get("cuda_available"),
        "cuda_version": hw.get("cuda_version"),
        "mps_available": hw.get("mps_available"),
        "gpu_model": hw.get("gpu_model"),
        "gpu_count": hw.get("gpu_count"),
        "device": hw.get("device"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timezone": time.tzname[0] if time.tzname else None,
    }
    return env


def _ram_gb() -> float | None:
    try:
        import resource
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return None  # self-rss is not system RAM; use sysctl below
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return round(int(out.stdout.strip()) / 1e9, 2)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1e6, 2)
    except Exception:
        pass
    return None
