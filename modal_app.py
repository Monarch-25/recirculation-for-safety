"""Modal GPU execution for Phase-2 GSM8K evals (scaffold — not yet run).

Usage (after `modal setup` + secrets, on a machine with modal installed):

    modal secret create huggingface HF_TOKEN=<paste from access_tokens.txt>
    modal run modal_app.py --config configs/gsm8k_gemma3_1b_pt_baseline_vllm.yaml --limit 5
    modal run modal_app.py --config configs/gsm8k_gemma3_1b_pt_baseline_vllm.yaml   # full 1319

Size parameterization is via the config file:
    1B  -> configs/gsm8k_gemma3_1b_pt_*_vllm.yaml   (default focus)
    4B  -> same layout with google/gemma-3-4b-pt + source/destination 18/9
    12B -> same layout with google/gemma-3-12b-pt + source/destination 35/16

GPU is fixed at decoration time (Modal requirement). A100-40GB fits all
three sizes under vLLM in bf16; downgrade to L4/A10G for 1B-only work by
editing GPU_TYPE below. Nothing here executes locally.
"""

from __future__ import annotations

import subprocess
import sys

import modal

GPU_TYPE = "A100-40GB"  # L4/A10G suffice for 1B-only iteration
RESULTS_VOLUME = "recirc-results"
HF_SECRET = "huggingface"  # must provide HF_TOKEN (gated Gemma access)

app = modal.App("recirc-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.50",
        "datasets>=2.18",
        "pyyaml>=6",
        "huggingface-hub>=0.20",
        "numpy",
        "vllm>=0.7",
    )
    .add_local_dir("src", remote_path="/root/recirc/src")
    .add_local_dir("scripts", remote_path="/root/recirc/scripts")
    .add_local_dir("configs", remote_path="/root/recirc/configs")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 6,  # full GSM8K headroom
    volumes={"/root/recirc/results": modal.Volume.from_name(
        RESULTS_VOLUME, create_if_missing=True)},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_eval(config: str, limit: int | None, batch_size: int | None) -> str:
    """Execute scripts/evaluate.py on the GPU worker; results persist."""
    cmd = [sys.executable, "scripts/evaluate.py", "--config", config]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd="/root/recirc")
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py failed (rc={proc.returncode})")
    # Last "Run artifacts:" line carries the run dir for convenience.
    tail = [l for l in proc.stdout.splitlines() if l.startswith("results/")]
    return tail[-1] if tail else "artifacts in /root/recirc/results (see logs)"


@app.local_entrypoint()
def main(config: str, limit: int | None = None,
         batch_size: int | None = None) -> None:
    run_dir = run_eval.remote(config, limit, batch_size)
    print(f"Done. {run_dir}")
    print("Sync back with: modal volume get recirc-results <run-path> ./results/")
