"""Capture instrumented cross-step traces (per-position norms + tokens).

Default invocation reproduces the frozen appendix trace byte-for-byte:
    conda run -n torch python scripts/trace_walkthrough.py \\
        --out reports/data/xstep_trace.json

Subset mode (real GSM8K examples, e.g. for Modal trajectory logs):
    python scripts/trace_walkthrough.py --examples dataset \\
        --model google/gemma-3-4b-pt --dtype bfloat16 --attn null \\
        --source 18 --dest 7 --alpha 0.1 --beta 1.0 \\
        --mixture-mode nonconvex --max-new-tokens 256 \\
        --start-index 0 --num-examples 20 --out traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_adapter(args, num_layers_hint=None):
    from eval_harness.models.recirculation import RecirculationModelAdapter

    attn = None if args.attn == "null" else args.attn
    return RecirculationModelAdapter(
        args.model, dtype=args.dtype, device="auto",
        attn_implementation=attn, use_chat_template=args.use_chat_template,
        debug_steps=args.debug_steps,
        intervention={"type": "recirculation",
                      "source_layer": args.source,
                      "destination_layer": args.dest, "alpha": args.alpha,
                      "beta": args.beta,
                      "mixture": {"mode": args.mixture_mode},
                      "normalization": {"type": "destination_l2"},
                      "ramping": {"enabled": args.ramp_tokens > 0,
                                  "tokens": args.ramp_tokens}})


def trace_one(adapter, task, ex, args):
    from eval_harness.core.interfaces import GenerationConfig

    prompt = task.build_prompt(ex)
    out = adapter.generate(
        [prompt],
        GenerationConfig(max_new_tokens=args.max_new_tokens,
                         do_sample=False, temperature=0.0, top_p=1.0,
                         seed=args.seed))[0]
    ids = adapter.tokenizer(prompt)["input_ids"]
    return {
        "model": adapter.model_id,
        "model_revision": adapter.model_revision,
        "layers": {"source": args.source, "destination": args.dest,
                   "total": adapter.get_metadata()["num_layers"]},
        "alpha": args.alpha,
        "beta": args.beta,
        "mixture": args.mixture_mode,
        "normalization": "destination_l2",
        "ramping": f"linear-{args.ramp_tokens}" if args.ramp_tokens else "off",
        "question": ex.question,
        "reference_answer": ex.reference_answer,
        "prompt": prompt,
        "input_tokens": adapter.tokenizer.convert_ids_to_tokens(ids),
        "n_prefill": len(ids),
        "n_generated": len(adapter.tokenizer(out)["input_ids"]),
        "generated_text": out,
        "steps": adapter.last_debug,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default="eager",
                    help="'null' = transformers default (SDPA on CUDA)")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--source", type=int, default=11)
    ap.add_argument("--dest", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--mixture-mode", default="convex")
    ap.add_argument("--ramp-tokens", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", default="fixture",
                    choices=["fixture", "dataset"])
    ap.add_argument("--example-index", type=int, default=1)
    ap.add_argument("--dataset-id", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default="main")
    ap.add_argument("--dataset-revision",
                    default="740312add88f781978c0658806c59bc2815b9866")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-examples", type=int, default=1)
    ap.add_argument("--debug-steps", type=int, default=400)
    ap.add_argument("--out", default="reports/data/xstep_trace.json")
    args = ap.parse_args(argv)

    from eval_harness.core.interfaces import EvalExample
    from eval_harness.tasks.gsm8k import GSM8KTask, make_example_id

    task = GSM8KTask(template_name="gsm8k_kojima_v1")
    if args.examples == "fixture":
        rows = json.load(open("tests/fixtures/gsm8k_tiny.json"))
        row = rows[args.example_index]
        examples = [EvalExample(
            make_example_id(args.example_index, row["question"]),
            args.example_index, row["question"], row["answer"])]
    else:
        task = GSM8KTask(
            template_name="gsm8k_kojima_v1",
            dataset_id=args.dataset_id, dataset_config=args.dataset_config,
            dataset_revision=args.dataset_revision)
        examples = task.load(args.dataset_split, limit=None)[
            args.start_index:args.start_index + args.num_examples]

    adapter = build_adapter(args)
    records = [trace_one(adapter, task, ex, args) for ex in examples]
    if len(records) == 1:
        Path(args.out).write_text(json.dumps(records[0], indent=1) + "\n")
    else:
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    print(f"traced {len(records)} example(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
