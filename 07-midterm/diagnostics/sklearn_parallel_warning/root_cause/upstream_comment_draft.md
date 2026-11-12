<!--
Draft comment for https://github.com/scikit-learn/scikit-learn/issues/35089
Not posted. The links point to the main branch, so merge the branch with the
root-cause work before posting. Delete this comment block when posting.
-->

I can reproduce this on a GIL build with an ordinary workload and the default switch interval, and I have a trace of the list objects involved.

**Setup.** Python 3.14.6 (GIL build), scikit-learn 1.9.1, joblib 1.6.0. `GridSearchCV(n_jobs=16)` over `RandomForestClassifier(n_estimators=200, n_jobs=16)`, 6 candidates, 5 folds, 8,844 rows and 30 features, on a 4-core machine. No `sys.setswitchinterval`. One fresh process tree per attempt.

**Rate.** The `delayed` / `Parallel` warning appeared in 21 of 30 attempts with a tracer loaded and in 5 of 10 without it. With `PYTHON_CONTEXT_AWARE_WARNINGS=1`: 0 of 30 and 0 of 10. With `RandomForestClassifier(n_jobs=1)` inside the same grid search: 0 of 30 and 0 of 10. It depends on the layout: with 4 worker processes of 4 threads each I saw none in 10 attempts, and none in 8 with a sequential grid search and 16 forest threads.

**Trace.** The tracer records, under `warnings._lock`, the `id()` and length of the list each `catch_warnings.__enter__`, `catch_warnings.__exit__` and `resetwarnings()` acts on, plus the list each `_FuncWrapper` task was handed. In the 30 baseline attempts (366,000 thread tasks, 1,830 thread-backed `Parallel` calls) all 9,190 warnings fall into two paths:

1. *The handed list is emptied during the call* (2,390 warnings, 20 calls). `Parallel.__call__` hands the tasks `warnings._get_filters()`, which is the live `warnings.filters` object. A thread enters `catch_warnings`, another thread's exit makes the handed list global again, and the first thread's `resetwarnings()` then empties it. All 21 recorded resets have that order. The remaining tasks of the call warn.
2. *The next call captures an empty list* (6,800 warnings, 34 calls). After `fit`, the global list is a copy created by a worker thread's `__enter__` and left behind by an out-of-order `__exit__`. `predict_proba`'s `Parallel.__call__` captures it and all 200 tasks warn.

A third outcome has no warning: the leftover list is shortened but not empty, so line 143 does not fire and the tasks run with filters missing (26 calls; 5,679 tasks ran with a shortened list).

**Copying the list is not enough.** With `Parallel.__call__` patched to hand tasks `list(warnings._get_filters())`, path 1 disappears (0 resets of a handed list in 30 attempts) but 12 of 30 attempts still warn through path 2, and 20 of 30 still run tasks with a damaged list.

**A user's own filter does not hide it.** With `warnings.filterwarnings("ignore", message=...)` set before `fit`, 4,759 of 5,451 raised warnings were still shown, because the list in force when the warning is raised is usually one of the damaged ones.

**Deterministic reproducer.** [This script](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py) forces path 1 with two threads and four trivial tasks by ordering the two threads' entry and exit. It needs no shortened switch interval and gives "11 -> 0 filters" and two warnings on every run.

Report, tracer, analyzer and raw traces of the affected processes: [ROOT_CAUSE.md](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/ROOT_CAUSE.md).
