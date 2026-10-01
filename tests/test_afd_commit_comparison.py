"""The paired gate must reject drift even when both executions finish."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_afd_commits", Path(__file__).resolve().parents[1] / "scripts/compare_afd_commits.py"
)
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


@pytest.fixture
def pair():
    query = {"afd_phase": "decode", "backend_version": "test"}
    case = {
        "id": "decode",
        "overrides": {},
        "metrics": {"tpot": 10.0},
        "path": {"phase": "decode"},
        "phases": ["decode"],
    }
    manifest = {"query": query, "cases": [case]}
    row = {
        "id": "decode",
        "query": query,
        "metrics": {"tpot": 10.0},
        "path": case["path"],
        "phases": ["decode"],
        "mode": "afd",
        "backend_version": "test",
    }
    snapshot = {"input_sha256": {"input": "hash"}, "cases": [row]}
    return deepcopy(snapshot), deepcopy(snapshot), deepcopy(manifest)


def test_identical_pair_passes(pair):
    result = _HARNESS.compare(*pair, 0)
    assert result["passed"]
    assert result["deltas"][0]["delta_ms"] == 0


@pytest.mark.parametrize(
    "mutation", ["slower", "faster", "missing", "duplicate", "empty-metrics", "nan", "path", "inputs", "query", "error"]
)
def test_gate_rejects_corruption(pair, mutation):
    before, after, manifest = pair
    row = after["cases"][0]
    if mutation in ("slower", "faster", "nan"):
        row["metrics"]["tpot"] = {"slower": 11.0, "faster": 9.0, "nan": float("nan")}[mutation]
    elif mutation == "missing":
        after["cases"] = []
    elif mutation == "duplicate":
        after["cases"].append(deepcopy(row))
    elif mutation == "empty-metrics":
        row["metrics"] = {}
    elif mutation == "path":
        row["path"]["phase"] = "prefill"
    elif mutation == "inputs":
        after["input_sha256"]["input"] = "changed"
    elif mutation == "query":
        # Matching but incorrect queries on BOTH sides must also fail.
        before["cases"][0]["query"]["extra"] = True
        row["query"]["extra"] = True
    elif mutation == "error":
        row["error"] = {"type": "RuntimeError", "message": "unexpected", "origin": "run"}
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]


def test_input_drift_requires_explicit_mode(pair):
    before, after, manifest = pair
    after["input_sha256"] = {"input": "changed"}
    result = _HARNESS.compare(before, after, manifest, 0, allow_input_drift=True)
    assert result["passed"]
    assert result["comparison_kind"] == "code-and-data"


def test_expected_errors_are_checked_on_both_sides(pair):
    before, after, manifest = pair
    manifest["cases"][0] = {
        "id": "decode",
        "overrides": {},
        "error": {"type": "ValueError", "match": "invalid phase", "origin": "cli_estimate"},
    }
    error = {"type": "ValueError", "message": "invalid phase: bad", "origin": "cli_estimate"}
    for snapshot in (before, after):
        snapshot["cases"][0] = {"id": "decode", "query": manifest["query"], "error": deepcopy(error)}
    assert _HARNESS.compare(before, after, manifest, 0)["passed"]
    for snapshot in (before, after):
        snapshot["cases"][0]["error"]["origin"] = "wrong_layer"
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]


def test_empty_manifest_is_not_a_pass(pair):
    before, after, manifest = pair
    manifest["cases"] = []
    before["cases"] = after["cases"] = []
    with pytest.raises(ValueError, match="nonempty"):
        _HARNESS.compare(before, after, manifest, 0)


def test_pd_combination_requires_complementary_metric(pair):
    before, after, manifest = pair
    manifest["query"]["afd_combined_with_pd"] = True
    for snapshot in (before, after):
        snapshot["cases"][0]["query"]["afd_combined_with_pd"] = True
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]
