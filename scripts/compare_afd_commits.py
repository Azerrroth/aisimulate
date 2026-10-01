"""Run the same AFD queries in two prepared checkouts and fail on drift.

Each checkout needs its own `uv sync --project python/aisimulate --extra dev`.
This harness never checks out revisions, installs dependencies or rewrites targets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def capture(repo, manifest, systems_path):
    # Select by files, not ImportError: a broken dependency must fail visibly.
    source = repo / "python/aisimulate/src"
    modern = (source / "aisimulate/legacy_cli/api.py").is_file()
    core = importlib.import_module("aisimulate_core" if modern else "aiconfigurator_core")
    api = importlib.import_module("aisimulate.legacy_cli.api" if modern else "aiconfigurator.cli.api")
    for module in (core, api):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"Wrong checkout imported: {module.__file__}")
    dirty = git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "python/aisimulate",
        "crates",
        "Cargo.toml",
        "Cargo.lock",
    )
    if dirty:
        raise RuntimeError(f"Runtime checkout must be clean: {dirty}")
    root = Path(core.__file__).parent
    system_root = systems_path or root / "systems"
    inputs = {}
    for relative in manifest["input_sha256"]:
        if not relative.startswith("systems/"):
            inputs[relative] = digest(root / relative)
    inputs["systems/h200_sxm.yaml"] = digest(system_root / "h200_sxm.yaml")
    for path in sorted((system_root / "data/h200_sxm").rglob("*")):
        if path.is_file():
            inputs["systems/" + str(path.relative_to(system_root))] = digest(path)
    rows = []
    for case in manifest["cases"]:
        query = {**manifest["query"], **case["overrides"]}
        if systems_path:
            query["systems_paths"] = str(systems_path)
        row = {"id": case["id"], "query": query}
        try:
            result = api.cli_estimate(**query)
            phase = query["afd_phase"]
            keys = {"prefill": ["ttft"], "decode": ["tpot"], "both": ["ttft", "tpot"]}[phase]
            if query.get("afd_combined_with_pd"):
                keys = ["ttft", "tpot"]
            row.update(
                metrics={key: getattr(result, key) for key in keys},
                path={key: result.raw[key] for key in case.get("path", manifest["cases"][0]["path"])},
                phases=sorted(result.raw["afd_layer_measurements"]),
                mode=result.mode,
                backend_version=result.backend_version,
            )
        except Exception as exc:
            row["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "origin": traceback.extract_tb(exc.__traceback__)[-1].name,
            }
        rows.append(row)
    return {
        "commit": git(repo, "rev-parse", "HEAD"),
        "repo": str(repo),
        "python": sys.version,
        "executable": sys.executable,
        "api": api.__file__,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "input_sha256": inputs,
        "cases": rows,
    }


def compare(before, after, manifest, tolerance, allow_input_drift=False):
    failures = []
    deltas = []
    expected_ids = [case["id"] for case in manifest["cases"]]
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Case manifest must be nonempty with unique IDs")
    inputs_equal = before["input_sha256"] == after["input_sha256"]
    if not inputs_equal and not allow_input_drift:
        failures.append("Input fingerprints differ: use common --systems-path for paired code regression")
    for label, snapshot in (("before", before), ("after", after)):
        if [row["id"] for row in snapshot["cases"]] != expected_ids:
            failures.append(f"{label}: missing, duplicate, reordered or unexpected cases")
    if failures and any("cases" in item for item in failures):
        return {
            "passed": False,
            "failures": failures,
            "deltas": [],
            "inputs_equal": inputs_equal,
            "comparison_kind": "paired-code" if inputs_equal else "code-and-data",
        }
    for case, old, new in zip(manifest["cases"], before["cases"], after["cases"], strict=True):
        name = case["id"]
        expected_query = {**manifest["query"], **case["overrides"]}
        for label, row in (("before", old), ("after", new)):
            if {k: v for k, v in row["query"].items() if k != "systems_paths"} != expected_query:
                failures.append(f"{name}/{label}: query differs from manifest")
        if old["query"] != new["query"]:
            failures.append(f"{name}: resolved queries differ")
        if "error" in case:
            target = case["error"]
            for label, row in (("before", old), ("after", new)):
                error = row.get("error", {})
                if (
                    error.get("type") != target["type"]
                    or error.get("origin") != target["origin"]
                    or not re.search(target["match"], error.get("message", ""))
                ):
                    failures.append(f"{name}/{label}: unexpected rejection {error}")
            continue
        if "error" in old or "error" in new:
            failures.append(f"{name}: unexpected execution error")
            continue
        phase = {**manifest["query"], **case["overrides"]}["afd_phase"]
        required = {"decode": {"tpot"}, "prefill": {"ttft"}, "both": {"ttft", "tpot"}}[phase]
        if expected_query.get("afd_combined_with_pd"):
            required = {"ttft", "tpot"}
        for label, row in (("before", old), ("after", new)):
            if set(row["metrics"]) != required:
                failures.append(f"{name}/{label}: wrong metric coverage")
            if row["path"] != case["path"] or row["phases"] != sorted(case["phases"]):
                failures.append(f"{name}/{label}: path or phase contract changed")
            if row["mode"] != "afd" or row["backend_version"] != row["query"]["backend_version"]:
                failures.append(f"{name}/{label}: wrong mode or backend version")
        for metric in required:
            left, right = old["metrics"].get(metric), new["metrics"].get(metric)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (left, right)):
                failures.append(f"{name}/{metric}: missing or invalid latency")
                continue
            relative = (right - left) / left
            deltas.append(
                {
                    "id": name,
                    "metric": metric,
                    "before": left,
                    "after": right,
                    "delta_ms": right - left,
                    "relative_delta": relative,
                }
            )
            if abs(relative) > tolerance:
                failures.append(f"{name}/{metric}: relative drift {relative:.9g} exceeds {tolerance}")
    return {
        "passed": not failures,
        "inputs_equal": inputs_equal,
        "comparison_kind": "paired-code" if inputs_equal else "code-and-data",
        "failures": failures,
        "deltas": deltas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--before-python", type=Path)
    parser.add_argument("--after-python", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "tests/e2e/afd_estimate_baseline.json"
    )
    parser.add_argument(
        "--systems-path", type=Path, help="Shared systems directory for identical-input code comparison"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--allow-input-drift", action="store_true", help="Explicit code-and-data comparison, not paired code regression"
    )
    parser.add_argument("--capture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.relative_tolerance) or args.relative_tolerance < 0:
        parser.error("relative tolerance must be finite and nonnegative")
    manifest = json.loads(args.manifest.read_text())
    systems = args.systems_path.resolve() if args.systems_path else None
    if args.capture:
        result = capture(args.capture.resolve(), manifest, systems)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    if not args.before or not args.after:
        parser.error("--before and --after are required")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    try:
        for label in ("before", "after"):
            repo = getattr(args, label).resolve()
            python = getattr(args, label + "_python") or repo / "python/aisimulate/.venv/bin/python"
            output = args.output / f"{label}.json"
            command = [
                str(python.absolute()),
                str(Path(__file__).resolve()),
                "--capture",
                str(repo),
                "--manifest",
                str(args.manifest.resolve()),
                "--output",
                str(output.resolve()),
            ]
            if systems:
                command += ["--systems-path", str(systems)]
            env = {**os.environ, "PYTHONPATH": str(repo / "python/aisimulate/src"), "PYTHONNOUSERSITE": "1"}
            with (args.output / f"{label}.log").open("w") as log:
                subprocess.run(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600
                )
            snapshots.append(json.loads(output.read_text()))
        result = compare(*snapshots, manifest, args.relative_tolerance, args.allow_input_drift)
        result.update(
            before_commit=snapshots[0]["commit"],
            after_commit=snapshots[1]["commit"],
            manifest_sha256=digest(args.manifest),
            harness_sha256=digest(__file__),
            relative_tolerance=args.relative_tolerance,
            case_count=len(manifest["cases"]),
        )
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        lines = [
            f"# AFD commit comparison: {'PASS' if result['passed'] else 'FAIL'}",
            "",
            f"Before: `{result['before_commit']}`",
            f"After: `{result['after_commit']}`",
            f"Cases: {result['case_count']}; kind: {result['comparison_kind']}; "
            f"relative tolerance: {args.relative_tolerance}",
            "",
            "| Case | Metric | Before ms | After ms | Delta ms | Delta % |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in result["deltas"]:
            lines.append(
                f"| {row['id']} | {row['metric']} | {row['before']} | {row['after']} | "
                f"{row['delta_ms']:.6f} | {row['relative_delta'] * 100:.6f} |"
            )
        lines += ["", *[f"- {failure}" for failure in result["failures"]]]
        (args.output / "report.md").write_text("\n".join(lines) + "\n")
        print(
            json.dumps(
                {key: result[key] for key in ("passed", "case_count", "inputs_equal", "comparison_kind", "failures")}
            )
        )
        return 0 if result["passed"] else 2
    except Exception as exc:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
