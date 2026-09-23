#!/usr/bin/env python
"""Entrypoint run by `jl run . --script scripts/jl_eval_entry.py --on <id>`.

Executes on the JarvisLabs instance inside the synced project's venv.
Loads secrets from scripts/.jl_secrets.env (JSON, written by jl_run.sh,
gitignored), forwards all extra args to scripts/evaluate.py, and packages
the produced run dir so jl_run.sh can `jl download` it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def main(argv: list[str]) -> int:
    secrets = ROOT / "scripts" / ".jl_secrets.env"
    if secrets.exists():
        os.environ.update(json.loads(secrets.read_text()))
        print(f"[jl] loaded secrets from {secrets}")

    print(f"[jl] project root: {ROOT}")
    try:
        import torch
        n = torch.cuda.device_count()
        print(f"[jl] torch {torch.__version__} cuda_available="
              f"{torch.cuda.is_available()} devices={n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"[jl]   [{i}] {p.name} {p.total_memory / 2**30:.1f} GiB")
    except Exception as exc:  # never gate the eval on this check
        print(f"[jl] torch probe failed (non-fatal): {exc}")

    print(f"[jl] evaluate.py args: {' '.join(argv) or '<none>'}")
    if "--config" not in argv:
        print("[jl] no --config in args; nothing to do", file=sys.stderr)
        return 2
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate.py", *argv],
        cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        print(f"[jl] evaluate FAILED (rc={proc.returncode})", file=sys.stderr)
        return proc.returncode

    run_dir = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("results/"):
            run_dir = line.strip()
            break
    if run_dir:
        package = Path(os.environ.get(
            "JL_RESULTS_PACKAGE", "/home/recirc_results.tar.gz"))
        tar = subprocess.run(["tar", "-czf", str(package), run_dir],
                             cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if tar.returncode == 0:
            print(f"[jl] results_package={package}")
        else:
            print(f"[jl] packaging failed: {tar.stdout}", file=sys.stderr)
    else:
        print("[jl] no results/ run dir found in output", file=sys.stderr)
    print("[jl] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))