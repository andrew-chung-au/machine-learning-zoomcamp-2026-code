<!--
Draft reply for https://github.com/DataTalksClub/faq/issues/551
Not posted. The links point to the main branch, so merge the branch with the
root-cause work before posting. Delete this comment block when posting.
-->

Update: I have now observed the cause directly, and the recommended fix in the entry has changed.

**Cause**

In every parallel task, scikit-learn 1.9.1 swaps Python's warning-filter list with `warnings.catch_warnings()`, empties it with `warnings.resetwarnings()` and re-adds the caller's filters (`sklearn/utils/parallel.py`, lines 154-182). Without context-aware warnings that list is shared by all threads in a process, and those calls are not thread-safe. `RandomForestClassifier(n_jobs=-1)` runs its tasks in threads, so the threads overwrite each other's swaps and a filter list ends up empty. The check on line 143 (`if not config or not warning_filters`) then reads the empty list as "`delayed` was used without `Parallel`".

So the warning is a false alarm in this case: `Parallel` and `delayed` are paired, and the configuration was present in every recorded warning.

**Minimal reproducer**

No dataset, no estimator, about one second, the same result on every run. It uses scikit-learn's own `Parallel` and `delayed` with two threads and four tasks, and fixes the order in which the two threads enter and leave `catch_warnings`:

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py
```

```text
Before: 11 warning filters; the list handed to the tasks is the live warnings.filters object: True
After:  0 warning filters in the list handed to the tasks; results=[1, 2, 3, 4]
Target warnings emitted: 2
  {'thread': 'Thread-1 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
  {'thread': 'Thread-2 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
```

With `PYTHON_CONTEXT_AWARE_WARNINGS=1` the same script emits no warning and the list keeps its 11 filters.

**Real workload**

I traced which list object every `catch_warnings` entry, exit and `resetwarnings()` acted on, in the main process and all joblib workers, during the same `GridSearchCV` over `RandomForestClassifier` as before. All 9,190 warnings in the baseline runs were traced to a recorded event in which a worker thread emptied a filter list that other tasks were using. None was left unexplained.

| Setup (fresh process per run) | Runs with the warning |
| --- | --- |
| Grid search and forest both parallel | 21 of 30 |
| Forest single-threaded, grid search still parallel | 0 of 30 |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 30 |
| Both parallel, with `warnings.filterwarnings("ignore", ...)` | 12 of 15 |

Fisher exact test for each of the two fixes against the first row: p = 3.6e-9. Without the tracer the first three rows were 5 of 10, 0 of 10 and 0 of 10.

**What changed in the entry**

- The recommended fix is now to parallelise at one level: `n_jobs=-1` on `GridSearchCV`, no `n_jobs` on the model. It works on any Python version, needs no restart, and was not slower in my tests.
- Context-aware warnings are the second option, for Python 3.14 or later.
- The `filterwarnings` step is now described as not working, with the reason: the filter is one of the entries that gets lost.
- The entry separates this case from the one where the message is accurate (your own code mixing `joblib` and scikit-learn's `Parallel` / `delayed`).

**Limits**

- The new runs are from one 4-core machine (Python 3.14.6, scikit-learn 1.9.1, joblib 1.6.0), set to 16 worker processes and 16 forest threads to match the layout of my original runs. With that machine's own `n_jobs=-1` (4 processes, 4 threads each) the warning did not appear in 10 runs, so the rate depends on the layout.
- One dataset and one parameter grid.
- The two options avoid the conditions the bug needs. The bug itself is in scikit-learn and is open as scikit-learn/scikit-learn#35089; the code on its `main` branch was the same as 1.9.1 when I checked on 11 November.

**Files**

- [Root-cause report](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/ROOT_CAUSE.md)
- [Revised entry](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/07-faq_pr.md)
- [Deterministic reproducer](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py)
- [Comparison runs and traces](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/tree/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause)
