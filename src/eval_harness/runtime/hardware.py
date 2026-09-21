"""Platform-aware hardware discovery.

Never assumes CUDA/nvidia-smi exist. Unavailable fields are None.
"""

from __future__ import annotations

import platform
from typing import Any


def get_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "mps_available": False,
        "gpu_model": None,
        "gpu_count": 0,
        "device": "cpu",
        "cpu": platform.processor() or None,
        "machine": platform.machine(),
    }
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        info["cuda_available"] = bool(cuda_ok)
        try:
            info["cuda_version"] = torch.version.cuda
        except Exception:
            info["cuda_version"] = None
        try:
            info["mps_available"] = bool(torch.backends.mps.is_available())
        except Exception:
            info["mps_available"] = False
        if cuda_ok:
            try:
                info["gpu_count"] = int(torch.cuda.device_count())
            except Exception:
                info["gpu_count"] = 0
            try:
                info["gpu_model"] = torch.cuda.get_device_name(0)
            except Exception:
                info["gpu_model"] = None
            info["device"] = "cuda"
        elif info["mps_available"]:
            info["gpu_count"] = 1
            info["gpu_model"] = _apple_gpu_model()
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
    except ImportError:
        pass
    return info


def _apple_gpu_model() -> str | None:
    """Best-effort Apple GPU label; None if undiscoverable."""
    import subprocess
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                s = line.strip()
                if s.startswith("Chipset Model:"):
                    return "Apple " + s.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Apple Silicon (MPS)"
