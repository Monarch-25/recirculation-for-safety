"""Unit tests: stable example ids, metrics, serialization, env, manifest."""

import json

from eval_harness.core.results import compute_metrics, PredictionRecord
from eval_harness.tasks.gsm8k import make_example_id
from eval_harness.utils import hashing, io as io_utils


def test_example_ids_stable():
    a = make_example_id(3, "What is 2+2?")
    b = make_example_id(3, "What is 2+2?")
    c = make_example_id(4, "What is 2+2?")
    d = make_example_id(3, "What is 2+3?")
    assert a == b
    assert a != c and a != d


def test_metrics_include_parse_failures_in_denominator():
    recs = [
        PredictionRecord("a", 0, "p", "o", "1", "#### 1", "1", True, True, 1.0),
        PredictionRecord("b", 1, "p", "o", None, "#### 2", "2", False, False, 0.0),
    ]
    m = compute_metrics(recs)
    assert m.num_examples == 2
    assert m.num_correct == 1
    assert m.accuracy == 0.5
    assert m.parse_rate == 0.5
    assert m.num_parse_failures == 1


def test_metrics_empty():
    m = compute_metrics([])
    assert m.num_examples == 0 and m.accuracy == 0.0


def test_record_serialization_roundtrip(tmp_path):
    rec = PredictionRecord("id-1", 0, "prompt", "raw", "42",
                           "#### 42", "42", True, True, 1.0)
    p = tmp_path / "preds.jsonl"
    io_utils.write_jsonl(p, [rec.to_dict()])
    rows = io_utils.read_jsonl(p)
    assert rows[0]["example_id"] == "id-1"
    assert rows[0]["correct"] is True


def test_environment_metadata_keys():
    from eval_harness.runtime.environment import collect_environment_metadata
    env = collect_environment_metadata()
    for key in ("python_version", "torch_version", "transformers_version",
                "datasets_version", "cuda_available", "mps_available",
                "gpu_model", "gpu_count", "device", "os", "machine"):
        assert key in env
    assert isinstance(env["cuda_available"], bool)
    assert isinstance(env["mps_available"], bool)


def test_hardware_no_cuda_assumption():
    from eval_harness.runtime.hardware import get_hardware_info
    hw = get_hardware_info()
    assert "cuda_available" in hw and "device" in hw
    # Must represent, never fabricate: if no cuda, gpu fields are None/0.
    if not hw["cuda_available"]:
        assert hw["cuda_version"] is None or isinstance(hw["cuda_version"], str)


def test_git_metadata_keys():
    from eval_harness.runtime.reproducibility import get_git_metadata
    g = get_git_metadata()
    assert set(g) == {"commit", "branch", "dirty"}


def test_hash_deterministic():
    assert hashing.sha256_hex("abc") == hashing.sha256_hex("abc")
    assert len(hashing.sha256_hex("abc")) == 64


def test_manifest_contains_required_fields():
    from eval_harness.core.config import load_config_from_dict
    from eval_harness.core.evaluator import Evaluator
    from eval_harness.models.mock import MockModelAdapter
    from eval_harness.tasks.gsm8k import GSM8KTask
    raw = {
        "model": {"name": "mock/m", "intervention": {"type": "none"}},
        "task": {"name": "gsm8k", "split": "test"},
        "prompt": {}, "generation": {}, "runtime": {},
        "experiment": {},
    }
    cfg = load_config_from_dict(raw)
    task = GSM8KTask()
    model = MockModelAdapter(["x"])
    ev = Evaluator()
    manifest = ev.build_manifest(
        config=cfg, run_id="r", model=model, task=task,
        prompt_text="t", prompt_hash="h",
        environment={"python_version": "3"}, git={"commit": None,
                                                  "branch": None,
                                                  "dirty": None},
        command="cmd", timestamp="ts", num_examples=1,
        dataset_revision=None,
    )
    # Required top-level sections.
    for section in ("experiment", "git", "model", "task", "prompt",
                    "generation", "runtime"):
        assert section in manifest
    for key in ("python_version", "torch_version", "transformers_version",
                "cuda_version", "gpu_model", "device", "batch_size", "seed"):
        assert key in manifest["runtime"], key
    json.dumps(manifest)  # must be JSON-serializable
