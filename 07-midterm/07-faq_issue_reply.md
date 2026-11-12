Update: A deterministic reproducer now demonstrates the warning-filter race, and tracing of the real workload supports the same explanation. Tests on two machines also show that both recommended workarounds eliminated the warning in the runs performed. The observed warning rate varies with execution conditions, so the results below distinguish traced from untraced runs.

I've also revised the FAQ entry to incorporate these findings and will post the full entry in a separate comment.

## Cause

In scikit-learn 1.9.1, the parallel wrapper in `sklearn/utils/parallel.py` uses `warnings.catch_warnings()`, calls `warnings.resetwarnings()`, and then restores the caller's warning filters (lines 154–182). Without context-aware warnings, the warning-filter list can be shared between threads, and these operations are not thread-safe.

When `RandomForestClassifier(n_jobs=-1)` runs tasks in threads, concurrent changes to that list can interfere with one another. The check on line 143 (`if not config or not warning_filters`) can then encounter an empty filter list and report that `delayed` was used without `Parallel`, even when the configuration is present and `Parallel` and `delayed` are correctly paired.

In the recorded warnings from this workload, the configuration was present. The evidence therefore supports this being a false alarm caused by the warning-filter race, rather than incorrect pairing of `Parallel` and `delayed`.

## Minimal reproducer

The reproducer runs without a dataset or estimator and takes about one second. It uses scikit-learn's own `Parallel` and `delayed` with two threads and four tasks, controlling the order in which the threads enter and leave `catch_warnings`.

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py
```

Output:

```text
Before: 11 warning filters; the list handed to the tasks is the live warnings.filters object: True
After:  0 warning filters in the list handed to the tasks; results=[1, 2, 3, 4]
Target warnings emitted: 2
  {'thread': 'Thread-1 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
  {'thread': 'Thread-2 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
```

With `PYTHON_CONTEXT_AWARE_WARNINGS=1`, the same script emits no warning and the list retains its 11 filters.

## Real workload

The traced runs help identify the race; the untraced runs show how often the warning appeared without the tracing instrumentation. Because instrumentation can affect thread timing, these rates should not be treated as directly interchangeable.

Every `catch_warnings` entry, exit, and `resetwarnings()` call was traced to the list object it acted on, across the main process and all joblib workers, during the same `GridSearchCV` over `RandomForestClassifier` as before.

| Setup (fresh process per run) | 16-core machine, `n_jobs=-1` | 4-core machine, 16 × 16 layout |
|:---|---:|---:|
| Grid search and forest both parallel | 13 of 15 warned; 15 of 15 corrupted | 21 of 30 warned; 30 of 30 corrupted |
| Forest single-threaded, grid search still parallel | 0 of 15 warned; 0 of 15 corrupted | 0 of 30 warned; 0 of 30 corrupted |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 15 warned; 0 of 15 corrupted | 0 of 30 warned; 0 of 30 corrupted |
| Both parallel, with `warnings.filterwarnings("ignore", ...)` | 11 of 15 warned; 14 of 15 corrupted | 12 of 15 warned |

In the traced baseline runs on both machines, all 12,690 observed warnings were accounted for by recorded events in which a worker thread emptied a warning-filter list shared with other tasks. This supports the race condition demonstrated by the deterministic reproducer as the cause of the warnings in these tests.

On the 16-core machine, the warning appeared in 13 of 15 traced baseline runs but in only 3 of 10 untraced runs. On the 4-core machine, it appeared in 21 of 30 traced runs and 5 of 10 untraced runs. This variation is consistent with an intermittent race whose observed frequency depends on execution conditions.

Neither the single-threaded-forest option nor context-aware warnings produced a target warning in the traced or untraced runs reported here.

Fisher exact tests comparing each recommended fix with the baseline traced run counts gave:

- 16-core machine: \(p = 1.8 \times 10^{-6}\)
- 4-core machine: \(p = 3.6 \times 10^{-9}\)

These results indicate a strong difference in the observed traced run counts. They do not establish the warning's frequency during ordinary, untraced execution.

### Untraced results

| Machine | Baseline | Context-aware warnings | Single-threaded forest |
|:---|---:|---:|---:|
| 16-core | 3 of 10 warned | 0 of 10 warned | 0 of 10 warned |
| 4-core | 5 of 10 warned | 0 of 10 warned | 0 of 10 warned |

### Performance and results

Median untraced times on the 16-core machine were:

| Setup | Median time |
|:---|---:|
| Baseline | 10.6 seconds |
| Context-aware warnings | 11.0 seconds |
| Single-threaded forest | 9.9 seconds |

With the tracer enabled, the single-threaded forest was also not slower in the 16-core tests: its median was 10.5 seconds versus 11.1 seconds for baseline.

The best score and best parameters were identical across all 60 traced attempts on the 16-core machine. These results suggest that using a single parallelisation level did not affect model selection in the tested workload; they do not establish that performance will be the same for other datasets or parameter grids.

## Recommended fixes

- **Recommended: parallelise at one level.** Set `n_jobs=-1` on `GridSearchCV` and leave `n_jobs` unspecified on the model. This avoids the nested parallel execution involved in the reproduction, works without requiring context-aware warnings, and was not slower in the measured workload.
- **Alternative: enable context-aware warnings.** Set `PYTHON_CONTEXT_AWARE_WARNINGS=1` when using Python 3.14 or later. This prevents the warning-filter interference observed in these tests.
- **Do not rely on `warnings.filterwarnings("ignore", ...)` as a fix.** The filter itself can be lost when the warning-filter list is emptied, and warnings continued to appear in the tested runs.

These workarounds eliminated the observed warnings in the runs reported here. The tests do not guarantee that either option will prevent every occurrence in every environment.

## What changed in the entry

- The primary recommendation is now to parallelise at one level: `n_jobs=-1` on `GridSearchCV`, with no `n_jobs` specified on the model.
- Context-aware warnings are presented as an alternative for Python 3.14 or later.
- The `filterwarnings` approach is described as unreliable, with an explanation of why the filter may be lost.
- The entry distinguishes this false alarm from cases where the warning is accurate, such as user code incorrectly mixing joblib and scikit-learn's `Parallel` / `delayed`.

## Limits

- **Two machines.** The 16-core machine used Python 3.14.5 and plain `n_jobs=-1`. The 4-core machine used Python 3.14.6 and a 16-process × 16-thread layout. Both used scikit-learn 1.9.1 and joblib 1.6.0.
- **The warning rate depends on execution conditions.** On the 4-core machine, its own `n_jobs=-1` configuration (4 processes, 4 forest threads each) did not produce the warning in 10 runs. The traced and untraced results also differed, so the observed rates should not be treated as universal estimates.
- **One dataset and one parameter grid.** More workloads and environments would be needed to generalise the performance findings.
- **Upstream status.** The underlying issue is tracked in [scikit-learn/scikit-learn#35089](https://github.com/scikit-learn/scikit-learn/issues/35089). When checked on 11 November, the relevant code on the project's `main` branch was the same as in version 1.9.1.

## Files

- [Root-cause report](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/ROOT_CAUSE.md)
- [Revised entry](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/07-faq_pr.md)
- [Deterministic reproducer](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py)
- [Comparison runs and traces](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/tree/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause)