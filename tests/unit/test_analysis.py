"""Unit tests: phase-2 analysis helpers (offline, Agg backend)."""

from analysis.gsm8k_phase2 import (
    mcnemar,
    pair_stats,
    transition_cells,
    wilson,
)


def _rows(spec):
    # spec: list of (base_correct, treat_correct)
    return [{"example_id": str(i), "baseline_correct": b,
             "treatment_correct": t, "baseline_chars": 10,
             "treatment_chars": 12, "baseline_parsed": "1",
             "treatment_parsed": "1"}
            for i, (b, t) in enumerate(spec)]


def test_cells_and_stats():
    rows = _rows([(1, 1), (1, 0), (0, 1), (0, 0)])
    s = pair_stats(rows)
    assert s["cells"] == {"cc": 1, "cw": 1, "wc": 1, "ww": 1}
    assert s["baseline_accuracy"] == 0.5
    assert s["absolute_delta"] == 0.0
    assert s["rescued"] == 1 and s["regressed"] == 1


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5
    lo, hi = wilson(10, 10)
    assert 0.5 < lo < 1.0 and hi == 1.0
    assert wilson(5, 0) == (0.0, 0.0)


def test_mcnemar_null_and_signal():
    chi2, p = mcnemar(0, 0)
    assert (chi2, p) == (0.0, 1.0)
    chi2, p = mcnemar(185, 163)
    assert abs(chi2 - 1.267) < 0.01
    assert abs(p - 0.260) < 0.01


def test_transition_cells_empty():
    assert transition_cells([]) == {"cc": 0, "cw": 0, "wc": 0, "ww": 0}


def test_figures_smoke(tmp_path):
    import csv
    from analysis import gsm8k_phase2 as A
    d = tmp_path / "data"
    d.mkdir()
    with open(d / "pair_1b_paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "baseline_correct", "treatment_correct",
                    "baseline_chars", "treatment_chars", "baseline_tokens",
                    "treatment_tokens", "baseline_parsed",
                    "treatment_parsed"])
        w.writerow(["e0", 1, 0, 50, 60, 10, 12, "1", "2"])
        w.writerow(["e1", 0, 0, 40, 45, 9, 9, "3", "3"])
    import json
    (d / "runs.json").write_text('{"pair_1b": {}, "pair_4b": {}}')
    import shutil
    shutil.copy(d / "pair_1b_paired.csv", d / "pair_4b_paired.csv")
    rows, _ = A.load_pair(d, "pair_1b")
    assert len(rows) == 2 and rows[0]["baseline_chars"] == 50
    for fig in (A.figure_a, A.figure_b, A.figure_c):
        out = fig(d, tmp_path / "f.png")
        assert out.exists()
    for fig in (A.figure_a, A.figure_b, A.figure_c):
        out = fig(d, tmp_path / "f1b.png", only="pair_1b")
        assert out.exists()


def test_load_sweep_and_figure_d(tmp_path):
    import csv
    from analysis import gsm8k_phase2 as A
    d = tmp_path / "data"
    d.mkdir()
    with open(d / "sweep_4b_100.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "alpha", "source_layer", "destination_layer",
                    "accuracy", "delta", "rescued", "regressed", "n",
                    "mcnemar_p", "baseline_accuracy", "baseline_n"])
        w.writerow(["a007_s18_d9", 0.07, 18, 9, 0.34, 0.09,
                    18, 9, 100, 0.124, 0.25, 100])
        w.writerow(["a015_s18_d9", 0.15, 18, 9, 0.29, 0.04,
                    15, 11, 100, 0.556, 0.25, 100])
    rows = A.load_sweep(d)
    assert len(rows) == 2 and rows[0]["alpha"] == 0.07
    out = A.figure_d(d, tmp_path / "figD.png")
    assert out.exists()
    out = A.figure_e(d, tmp_path / "figE.png")
    assert out.exists()


def test_sweep_transitions_and_figure_f(tmp_path):
    import csv
    from analysis import gsm8k_phase2 as A
    d = tmp_path / "data"
    d.mkdir()
    with open(d / "sweep_4b_100_transitions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "cc", "cw", "wc", "ww"])
        w.writerow(["a007_s18_d9", 16, 9, 18, 57])
        w.writerow(["a015_s18_d9", 14, 11, 15, 60])
    rows = A.load_sweep_transitions(d)
    assert len(rows) == 2 and rows[0]["wc"] == 18
    out = A.figure_f(d, tmp_path / "figF.png")
    assert out.exists()


def test_figure_g_requires_complete_slice(tmp_path):
    import csv
    from analysis import gsm8k_phase2 as A
    import pytest
    d = tmp_path / "data"
    d.mkdir()
    header = ["tag", "alpha", "source_layer", "destination_layer",
              "accuracy", "delta", "rescued", "regressed", "n",
              "mcnemar_p", "baseline_accuracy", "baseline_n"]
    with open(d / "sweep_4b_100.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in (16, 18, 20):
            for t, dst in enumerate((7, 9)):
                w.writerow([f"a010_s{s}_d{dst}", 0.10, s, dst,
                            0.30 + 0.01 * t, 0.05, 12, 7, 100, 0.4,
                            0.25, 100])
    out = A.figure_g(d, tmp_path / "figG.png")
    assert out.exists()
    # Holes must fail loudly, never render misleadingly: rows at
    # (18,9)+(16,7) imply a 2x2 grid missing two corners.
    with open(d / "sweep_4b_100.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(["a010_s18_d9", 0.10, 18, 9, 0.33, 0.08,
                    16, 8, 100, 0.15, 0.25, 100])
        w.writerow(["a010_s16_d7", 0.10, 16, 7, 0.33, 0.08,
                    16, 8, 100, 0.15, 0.25, 100])
    with pytest.raises(ValueError, match="missing cells"):
        A.figure_g(d, tmp_path / "figG2.png")
