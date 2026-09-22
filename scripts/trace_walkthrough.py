"""Capture an instrumented cross-step trace for the appendix walkthrough.

Runs one GSM8K fixture example through RecirculationModelAdapter with
debug tracing on and freezes per-position norms + token strings. Used
once to produce reports/data/xstep_trace.json; rerun to refresh it.

Usage:
    conda run -n torch python scripts/trace_walkthrough.py \\
        --out reports/data/xstep_trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--source", type=int, default=11)
    ap.add_argument("--dest", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--example-index", type=int, default=1)
    ap.add_argument("--debug-steps", type=int, default=400)
    ap.add_argument("--out", default="reports/data/xstep_trace.json")
    args = ap.parse_args(argv)

    from eval_harness.core.interfaces import EvalExample, GenerationConfig
    from eval_harness.models.recirculation import RecirculationModelAdapter
    from eval_harness.tasks.gsm8k import GSM8KTask, make_example_id

    rows = json.load(open("tests/fixtures/gsm8k_tiny.json"))
    row = rows[args.example_index]
    ex = EvalExample(make_example_id(args.example_index, row["question"]),
                     args.example_index, row["question"], row["answer"])
    task = GSM8KTask(template_name="gsm8k_kojima_v1")
    prompt = task.build_prompt(ex)

    adapter = RecirculationModelAdapter(
        args.model, dtype="float32", device="auto",
        attn_implementation="eager", use_chat_template=False,
        debug_steps=args.debug_steps,
        intervention={"type": "recirculation",
                      "source_layer": args.source,
                      "destination_layer": args.dest, "alpha": args.alpha,
                      "mixture": {"mode": "convex"},
                      "normalization": {"type": "destination_l2"},
                      "ramping": {"enabled": False}})
    out = adapter.generate(
        [prompt],
        GenerationConfig(max_new_tokens=args.max_new_tokens,
                         do_sample=False, temperature=0.0, top_p=1.0,
                         seed=42))[0]
    ids = adapter.tokenizer(prompt)["input_ids"]
    trace = {
        "model": adapter.model_id,
        "model_revision": adapter.model_revision,
        "layers": {"source": args.source, "destination": args.dest,
                   "total": adapter.get_metadata()["num_layers"]},
        "alpha": args.alpha,
        "beta": 0.85,
        "mixture": "convex",
        "normalization": "destination_l2",
        "ramping": "off",
        "question": row["question"],
        "reference_answer": row["answer"],
        "prompt": prompt,
        "input_tokens": adapter.tokenizer.convert_ids_to_tokens(ids),
        "n_prefill": len(ids),
        "n_generated": len(adapter.tokenizer(out)["input_ids"]),
        "generated_text": out,
        "steps": adapter.last_debug,
    }
    Path(args.out).write_text(json.dumps(trace, indent=1) + "\n")
    print(f"trace: {len(ids)} prefill tokens, "
          f"{len(adapter.last_debug)} debug steps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
