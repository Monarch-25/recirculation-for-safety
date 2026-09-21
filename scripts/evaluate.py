"""CLI: python scripts/evaluate.py --config <yaml> [--limit N] [--batch-size N] [--dry-run]."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_harness.core.config import load_config_from_yaml
from eval_harness.core.evaluator import Evaluator
from eval_harness.models.hf_causal_lm import HFCausalLMAdapter, resolve_device
from eval_harness.runtime import reproducibility
from eval_harness.runtime.environment import collect_environment_metadata
from eval_harness.tasks.gsm8k import GSM8KTask
from eval_harness.utils.logging import setup_logging

log = logging.getLogger(__name__)


def build_task(config):
    if config.task.name != "gsm8k":
        raise ValueError(f"Unknown task: {config.task.name!r}")
    return GSM8KTask(
        template_name=config.prompt.template_name,
        template_version=config.prompt.template_version,
        dataset_id=config.task.dataset_id,
        dataset_config=config.task.dataset_config,
        dataset_revision=config.task.dataset_revision,
        extraction_template_name=config.prompt.extraction_template_name,
        extraction_template_version=config.prompt.extraction_template_version,
    )


def build_model(config):
    want_recirc = config.model.intervention.type == "recirculation"
    if config.model.backend == "vllm":
        if want_recirc:
            raise ValueError(
                "intervention.type='recirculation' is not supported by the "
                "vLLM backend yet (needs engine-level support); use the "
                "default hf backend with RecirculationModelAdapter")
        from eval_harness.models.vllm_adapter import VLLMModelAdapter
        return VLLMModelAdapter(
            config.model.name,
            revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            dtype=config.model.dtype,
            device=config.model.device,
            trust_remote_code=config.model.trust_remote_code,
            use_chat_template=config.model.use_chat_template,
            tensor_parallel_size=config.model.tensor_parallel_size,
            gpu_memory_utilization=config.model.gpu_memory_utilization,
            enforce_eager=config.model.enforce_eager,
            max_model_len=config.model.max_model_len,
            seed=config.generation.seed,
            intervention=config.model.intervention,
        )
    if config.model.backend != "hf":
        raise ValueError(f"Unknown model backend: {config.model.backend!r}")
    if want_recirc:
        from eval_harness.models.recirculation import (
            RecirculationModelAdapter,
        )
        return RecirculationModelAdapter(
            config.model.name,
            revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            dtype=config.model.dtype,
            device=config.model.device,
            trust_remote_code=config.model.trust_remote_code,
            use_chat_template=config.model.use_chat_template,
            attn_implementation=config.model.attn_implementation,
            intervention=config.model.intervention,
        )
    return HFCausalLMAdapter(
        config.model.name,
        revision=config.model.revision,
        tokenizer_revision=config.model.tokenizer_revision,
        dtype=config.model.dtype,
        device=config.model.device,
        trust_remote_code=config.model.trust_remote_code,
        use_chat_template=config.model.use_chat_template,
        attn_implementation=config.model.attn_implementation,
        intervention=config.model.intervention,
    )


def print_header(config, device: str, num_examples: str | int) -> None:
    print("=" * 60)
    print("LLM Evaluation Harness")
    print("=" * 60)
    print(f"Task:                {config.task.name}")
    print(f"Task version:        {config.task.version}")
    print(f"Split:               {config.task.split}")
    print(f"Examples:            {num_examples}")
    print()
    print(f"Model:               {config.model.name}")
    print(f"Model revision:      {config.model.revision or '<to resolve>'}")
    print(f"Tokenizer revision:  {config.model.tokenizer_revision or '<to resolve>'}")
    print()
    print(f"Device:              {device}")
    print(f"DType:               {config.model.dtype}")
    print()
    print("Generation:")
    print(f"  temperature:       {config.generation.temperature}")
    print(f"  top_p:             {config.generation.top_p}")
    print(f"  do_sample:         {str(config.generation.do_sample).lower()}")
    print(f"  max_new_tokens:    {config.generation.max_new_tokens}")
    print(f"  seed:              {config.generation.seed}")
    print()
    print(f"Batch size:          {config.runtime.batch_size}")
    git = reproducibility.get_git_metadata()
    print(f"Git commit:          {git.get('commit')}")
    print(f"Git dirty:           {git.get('dirty')}")
    print(f"Output:              {config.runtime.output_dir}/"
          f"{config.task.name}/...")
    print("-" * 60, flush=True)


def cmd_dry_run(config, args) -> int:
    setup_logging()
    device = resolve_device(config.model.device)
    print_header(config, device, config.task.limit or "full split")
    print("Dry run checks:")
    env = collect_environment_metadata()
    print(f"  [ok] environment metadata collected "
          f"(python={env['python_version']} torch={env['torch_version']} "
          f"transformers={env['transformers_version']})")
    print(f"  [ok] device resolved: {device} "
          f"(cuda={env['cuda_available']} mps={env['mps_available']})")
    task = build_task(config)
    print(f"  [ok] task resolved: {task.name} v{task.version}")
    # Validate prompt generation on one synthetic example.
    from eval_harness.core.interfaces import EvalExample
    demo = EvalExample(example_id="dry-run-00000", index=0,
                       question="What is 2 + 2?",
                       reference_answer="#### 4")
    prompt = task.build_prompt(demo)
    assert "2 + 2" in prompt
    print("  [ok] prompt rendering works "
          f"(template={config.prompt.template_name} "
          f"v{config.prompt.template_version})")
    # Validate output dir writable.
    out = Path(config.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  [ok] output directory writable: {out.resolve()}")
    print("Dry run passed. No generation performed.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override task.limit (number of examples)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override runtime.batch_size")
    parser.add_argument("--device", type=str, default=None,
                        help="Override model.device (auto|cpu|mps|cuda)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config/env without generation")
    args = parser.parse_args(argv)

    config = load_config_from_yaml(
        args.config,
        limit_override=args.limit,
        batch_size_override=args.batch_size,
    )
    if args.device:
        import dataclasses
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model,
                                              device=args.device))
    if args.dry_run:
        return cmd_dry_run(config, args)

    setup_logging()
    device = resolve_device(config.model.device)
    print_header(config, device, config.task.limit or "full split")

    task = build_task(config)
    model = build_model(config)
    # Guardrail: warn if the model is far above the <500M bedroom.
    try:
        n = model.num_parameters
        print(f"Model parameters:    {n} "
              f"({'OK (<500M)' if n < 500_000_000 else 'WARNING: >=500M'})")
        if n >= 500_000_000:
            log.warning("model has %d params (>=500M budget)", n)
    except Exception:
        pass

    evaluator = Evaluator()
    result = evaluator.evaluate(
        task=task, model=model, config=config,
        command=" ".join(sys.argv),
    )
    m = result.metrics
    print("-" * 60)
    print("Results")
    print("-" * 60)
    print(f"Examples:            {m.num_examples}")
    print(f"Correct:             {m.num_correct}")
    print(f"Accuracy:            {m.accuracy:.4f}")
    print(f"Parse rate:          {m.parse_rate:.4f}")
    print()
    print(f"Run artifacts:\n{result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
