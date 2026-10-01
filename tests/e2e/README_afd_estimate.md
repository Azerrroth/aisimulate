# AFD estimate regression

This CPU-only suite runs fixed queries through the real
`aisimulate.legacy_cli.api.cli_estimate` entry point and compares public latency
outputs to frozen targets. It uses the native runtime and bundled performance
database without mocking estimator results.

From the repository root:

```bash
uv sync --project python/aisimulate --extra dev
python/aisimulate/.venv/bin/python -m pytest -p no:timeout \
  tests/e2e/test_afd_estimate_regression.py \
  --basetemp=/tmp/afd-estimate-regression --junitxml=/tmp/afd-estimate-results.xml
```

Use a dedicated `--basetemp` directory: pytest clears it on each run. Each
executed query writes a JSON record there with the resolved query, baseline
commit, input commit, target, actual, signed difference, relative difference, path outputs,
and captured logs. Pytest reports preflight failures separately.

## Configuration and coverage

`afd_estimate_baseline.json` pins Qwen/Qwen3-235B-A22B-FP8 on H200 with SGLang
0.5.14 in HYBRID mode. The common configuration uses one eight-GPU attention
node, one eight-GPU FFN node, attention TP=1, FFN EP=8, three microbatches,
ISL=16000, OSL=64 and six requests per attention worker.

The suite contains 28 query cases (21 numerical and seven rejection cases)
plus one manifest-integrity test:

- Conservative, optimistic and serial decode.
- Uneven microbatches and optimistic fallback with two microbatches.
- Prefill and both phases.
- Microbatch counts 1 and 4, attention TP 2 and 4, FFN EP 4, and two-node A/F pools.
- Short/long context, communication factors 0.5/1, FFN boundary assignment, and PD combinations.
- Invalid pipeline/phase, incompatible both-phase/PD, missing A/F node counts,
  zero microbatches, and zero attention TP.

Standalone decode compares TPOT, prefill compares TTFT, and both checks each
separately. PD combinations check both metrics, complementary static/AFD paths,
and the 24-GPU total budget.
Targets must be positive and finite. Public SDK latencies are rounded to three
decimal milliseconds, and the tests compare that output using `actual == target`.
They also check phase coverage, pipeline metadata, communication hiding,
microbatch size, worker topology, and rejection type/message/origin.

The integrity test rejects missing, duplicate or unexpected case IDs and
missing required metrics. Individual numerical cases also reject an empty or
phase-inappropriate metrics dictionary. Model configuration, system YAML and
H200 database files are checked against SHA-256 fingerprints, including the
database file inventory.

## Baseline lifecycle and scope

The top-level `commit` produced the original nine targets; each added case
records its own `target_commit` (`69165795` for the 19 added cases).
Independent subsequent runs verify deterministic regression behavior. Tests never
regenerate targets.
Changes to model/database inputs or targets require explicit review and a new
baseline capture. The current commit field records baseline provenance; it does
not select or check out that revision during execution.

This is a 28-query matrix for the current API. It is not a claim of one-to-one
coverage of an external design matrix. It does not implement measured per-step
accuracy comparison or attribution analysis. Passing these tests
does not establish agreement with measured GPU latency.

The repository Full CI contracts shard scans root `tests/`, so this suite is
eligible for collection once committed. Dedicated persistence of the per-case
JSON files is not configured here. The baseline records the capture Python
version but does not enforce a complete dependency or runtime fingerprint.

## Upstream compatibility refresh

The suite is rebased onto upstream `cab77b9c7888b1c40df8cd3228f5e03c355ef23c`.
Imports use the renamed `aisimulate_core` package and the retained
`aisimulate.legacy_cli.api.cli_estimate` entry point. This suite continues to
exercise that analytical AFD API.

`commit` preserves the original numerical-target provenance;
`input_commit` records the revision used for the refreshed input fingerprints.
No numerical targets, path expectations, error expectations or query settings
were changed during this refresh. The H200 inventory gained 36 files and lost
two FPM files; seven retained inputs changed (system YAML and GEMM/MLA data
or metadata). All 204 current input files remain fingerprinted.

Validation on the refreshed inputs: 10 tests passed on Python 3.13.7,
including all seven numerical cases and both rejection cases.

## Commit-to-commit comparison

`scripts/compare_afd_commits.py` executes this SAME manifest in two independently
prepared checkouts. It supports both the pre-rename and current Python packages.
Prepare each checkout with `uv sync --project python/aisimulate --extra dev`.
Runtime source/configuration must be clean; the script verifies the imported
Python module paths. It neither switches branches nor installs environments.

```bash
python scripts/compare_afd_commits.py \
  --before /absolute/path/to/before-checkout \
  --after /absolute/path/to/after-checkout \
  --systems-path /absolute/path/to/shared/systems \
  --output /absolute/path/to/new-comparison-directory
```

`--systems-path` is the directory containing `h200_sxm.yaml` and `data/`.
Use a version-controlled, unchanged directory for both runs. Without this option,
each checkout uses its bundled systems. The default gate rejects different model
or H200 input fingerprints. `--allow-input-drift` explicitly selects a combined
code-and-data comparison, which must not be interpreted as isolated code impact.
Separate Python executables can be selected with `--before-python` and
`--after-python`; otherwise each checkout's project venv is used. Interpreter
versions and installed package versions are recorded, but dependency equality is
not enforced: attribution to code alone requires reviewing those records too.

The paired gate checks all 28 resolved queries, required metrics, positive finite
latencies, phase/path metadata and rejection type/message pattern/origin on BOTH
sides. It compares the two executions directly; it does not compare each side to
a regenerated golden file. Fixed-target verification remains the separate pytest
suite above. The default relative tolerance is zero; any numerical change fails,
including faster results. `--relative-tolerance` is a fractional absolute relative
change bound (e.g. `0.01` = 1%) chosen before a run and recorded in its report.

Every output directory must be new. Outputs are `before.json`, `after.json`, both
execution logs, `comparison.json`, and `report.md` with per-metric before/after,
signed millisecond delta and percentage delta. Commit IDs, input hashes, query
coverage and harness/manifest hashes are retained. Execution/setup failures write
`failure.json` and return nonzero; completed comparisons return 0 for pass or 2
for gate failure. Failures are never dropped from the denominator.

The comparator has 15 additional guard tests in
`tests/test_afd_commit_comparison.py`; these are harness tests, not extra workload
cases. Pair execution is an explicit command, not an automatic CI job comparing
arbitrary commits. Existing CI can collect the 29 fixed-baseline tests and the 15
harness tests.

### Verified run (2026-10-01)

Compared `36b5465faf9aa383223533580006f1e387225c45` against
`6916579586561f22fbde2f70a7f3d395c2ec5a96` using the latter checkout's
H200 systems directory for both executions: 28/28 cases passed, all 24 numerical
comparisons had zero delta, and all seven expected rejections matched.
Python was 3.13.7 on both sides. Environment differences were AISimulate
0.12.0/0.13.0, googleapis-common-protos 1.75.1/1.75.3, and ijson absent/3.5.1;
this establishes observed agreement for these environments, not isolated causal
attribution to source changes. Using each checkout's own bundled database was
correctly rejected by the default input-equality gate (exit 2).

The fixed-baseline and harness tests together passed 44/44 in 3.04 seconds
(28 workloads + one integrity check + 15 comparator guard tests).
