# Root-cause tools

These tools establish why scikit-learn raises

```text
UserWarning: `sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel` ...
```

when the code only uses `GridSearchCV` and `RandomForestClassifier`, and they
test candidate fixes. The findings are in [`../ROOT_CAUSE.md`](../ROOT_CAUSE.md).

Installed scikit-learn is never modified. Run the commands from the repository
root.

| File | Purpose | Run time |
| --- | --- | --- |
| `deterministic_reproducer.py` | Smallest reproducer: scikit-learn's own `Parallel` and `delayed`, two threads, four tasks, fixed thread order. No dataset, no estimator, no randomness | about 1 second |
| `trace/filter_identity_trace.py`, `trace/sitecustomize.py` | Tracer for the real workload. Records which list object every `catch_warnings` entry, `catch_warnings` exit and `resetwarnings()` acts on, in the main process and in joblib workers | adds a little work per task |
| `analyze_trace.py` | Finds the event that empties the filter list in a set of traces, checks it against the expected order of events, and accounts for every warning | seconds |
| `compare_fixes.py` | Runs the real grid search under four settings (baseline, context-aware warnings, single-threaded forest, snapshot test patch), traced and analysed, and compares them | about 13 seconds per attempt here |
| `example-trace/` | Raw trace of one affected worker process from the comparison run, kept as a worked example | |
| `runs/` | Evidence from `compare_fixes.py` (one folder per run, never overwritten) | |
| `controls/` | Evidence from the control runs described in `ROOT_CAUSE.md` | |

## Commands

Deterministic reproducer (each mode checks its own expected outcome and exits
non-zero on a mismatch):

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py
```

```bash
PYTHON_CONTEXT_AWARE_WARNINGS=1 uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py
```

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py --snapshot
```

Comparison of fixes on the real workload. `-1` (the default) uses all cores.
Explicit numbers give a small machine the layout of a larger one:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/compare_fixes.py --rounds 15 --grid-n-jobs 16 --forest-n-jobs 16
```

Analysis of the kept example trace, with the sequence of events printed:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/analyze_trace.py 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/example-trace --show 1
```

Tracing a run by hand:

```bash
FILTER_TRACE_DIR=/tmp/filter-trace PYTHONPATH=07-midterm/diagnostics/sklearn_parallel_warning/root_cause/trace uv run python 07-midterm/reproduce_sklearn_parallel_warning.py --grid-n-jobs -1 --forest-n-jobs -1
```

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/analyze_trace.py /tmp/filter-trace
```

## Reading the tracer output

One JSON record per operation, in the order the operations took
`warnings._lock`:

| `k` | Operation | Fields |
| --- | --- | --- |
| `E` | `catch_warnings.__enter__` | `saved` (list that was global), `installed` (the copy made global), `handed` (list the scikit-learn task was given by `Parallel.__call__`, if the caller is the task wrapper) |
| `R` | `warnings.resetwarnings()` | `list` (list emptied), `len_before` |
| `X` | `catch_warnings.__exit__` | `restored` (list made global again) |
| `W` | the target warning | `handed`, `handed_len`, `current` (global list), `config_present` |

Lists are identified by `id()`. The analyzer only compares identities inside
one inner `Parallel` call, while the handed list is still referenced by that
call's tasks.

## Limits

- The tracer adds work to every task and takes `warnings._lock` around each
  operation, so it can change how often the race happens. It cannot change
  which list an operation acts on. The untraced control in `ROOT_CAUSE.md`
  shows the warning also occurs without it.
- With `PYTHON_CONTEXT_AWARE_WARNINGS=1` filters are stored per context, so
  "the list" in a record is the list of the calling thread's context. The
  first comparison run (`runs/20261111T213534Z`) used an earlier tracer that
  recorded only the warning itself in that mode; its `context_aware` row
  therefore measures warnings but not filter-list corruption.
- `--snapshot` and `FILTER_TRACE_SNAPSHOT=1` are test arms for a possible
  upstream change. They are not advice for users.
