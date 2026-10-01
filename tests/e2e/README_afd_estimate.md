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

The suite contains nine query cases plus one manifest-integrity test:

- Conservative, optimistic and serial decode.
- Uneven microbatches and optimistic fallback with two microbatches.
- Prefill and both phases.
- Invalid pipeline mode and an incompatible both-phase/PD combination.

Decode compares TPOT, prefill compares TTFT, and both checks each separately.
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

The recorded commit produced the initial targets; independent subsequent runs
verify deterministic regression behavior. Tests never regenerate targets.
Changes to model/database inputs or targets require explicit review and a new
baseline capture. The current commit field records baseline provenance; it does
not select or check out that revision during execution.

This is a nine-query regression subset. It does not implement the complete
28-query AFD branch matrix, measured per-step accuracy comparison, automatic
old/new checkout orchestration, or attribution analysis. Passing these tests
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
