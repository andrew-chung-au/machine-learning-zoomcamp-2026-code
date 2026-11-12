# scikit-learn parallel warning diagnostic suite

> **Where to start:** [`ROOT_CAUSE.md`](ROOT_CAUSE.md) has the confirmed cause,
> the deterministic reproducer and the comparison of fixes. The tools behind it
> are in [`root_cause/`](root_cause/README.md). [`FINDINGS.md`](FINDINGS.md) and
> `INVESTIGATION.md` are the earlier stages. The components described below are
> the first capture suite.

This suite investigates:

`sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel`

It uses the existing `07-midterm/reproduce_sklearn_parallel_warning.py`
as the workload. It does not patch installed scikit-learn source.

## Components

- `capture.py`: records matching `warnings.warn` calls in JSONL, including
  process ID, stack, captured config, and warning-filter state.
- `sitecustomize.py`: installs capture in Python processes that inherit the
  suite directory in `PYTHONPATH`, including joblib workers when they import
  site customization.
- `run_matrix.py`: runs all four outer/inner `n_jobs` combinations, with
  independent attempt directories and manifests.
- `inspect_environment.py`: records package versions, Git state, input hashes,
  and the installed parallel wrapper's hash and relevant source excerpts.
- `validate_evidence.py`: checks record completeness and consistency.

## Run

From the repository root, after activating the project's environment:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/run_matrix.py --repeat 3
```

This is the first model-running command. It creates a new timestamped folder
under `diagnostics/sklearn_parallel_warning/runs/`. Existing run folders and
historical logs are not overwritten.

After the runner finishes, use the exact path it prints:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/validate_evidence.py \
  07-midterm/diagnostics/sklearn_parallel_warning/runs/REPLACE_WITH_RUN_ID
```

Each attempt contains a manifest, combined stdout/stderr, and one JSONL capture
file per observed process.

## Interpretation

- `warning_observed`: one or more matching warning calls were captured.
- `no_target_warning_observed_capture_active`: the model process completed and
  at least one capture startup record exists, with no matching calls captured.
- `inconclusive_capture_not_verified`: no process reported capture startup.
- `failed`: the reproducer exited nonzero.

A capture-start record in any process does not prove every worker was captured.
Inspect each attempt's process capture files and logs before treating a
no-warning result as strong negative evidence. Capture instruments Python's
`warnings.warn`; it is not a general-purpose detector for warnings emitted
outside that mechanism.

A warning count is a count of captured warning calls, not independent runs.
A finite matrix cannot prove that a configuration never emits the warning or
establish a root cause on its own.

## Routine regression tests

From the repository root, run:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/run_tests.py
```

These checks use synthetic warnings and do not fit the model. They test
warning capture in joblib workers, filter-list clearing and replacement
detection, Python syntax, the deterministic reproducer in its three modes,
and the trace analyzer on the kept example trace. Successful test output is kept concise;
failure output is shown.

For additional filter diagnostics, set `SKLEARN_WARNING_TRACE_FILTERS=1`
when running a reproduction. The trace records in-place transitions from
a non-empty filter list to empty and flags filter-list replacement when a
target warning is observed. It does not capture every possible filter
change or establish the cause of the real warning.
