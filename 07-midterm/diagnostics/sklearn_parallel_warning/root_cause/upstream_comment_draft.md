> **Draft comment for upstream:** post on
> [scikit-learn/scikit-learn#35089](https://github.com/scikit-learn/scikit-learn/issues/35089)
> ("BUG: thread-based `Parallel` can corrupt the process-wide warning filters").

---

I can trigger this on a GIL build with an ordinary workload (`GridSearchCV` over `RandomForestClassifier`) and the default switch interval, so it does not need free-threaded Python or a shortened `sys.setswitchinterval`. I traced the list objects involved. It is intermittent, and I have tested two machines and one dataset (limits at the end).

**Setup.** Python 3.14.6 (GIL build), scikit-learn 1.9.1, joblib 1.6.0. `GridSearchCV(n_jobs=16)` over `RandomForestClassifier(n_estimators=200, n_jobs=16)`, 6 candidates, 5 folds, 8,844 rows and 30 features, on a 4-core machine. No `sys.setswitchinterval`. One fresh process tree per attempt.

**Rate.** The `delayed` / `Parallel` warning appeared in 21 of 30 attempts with a tracer loaded and in 5 of 10 without it. With `PYTHON_CONTEXT_AWARE_WARNINGS=1`: 0 of 30 and 0 of 10. With `RandomForestClassifier(n_jobs=1)` inside the same grid search: 0 of 30 and 0 of 10. On a 16-core machine (Python 3.14.5) with plain `n_jobs=-1` for both levels: 13 of 15 with a tracer loaded and 3 of 10 without it, against 0 of 15 and 0 of 10 for each of the two changes. The rate depends on the layout: with 4 worker processes of 4 threads each I saw none in 10 attempts, and none in 8 with a sequential grid search and 16 forest threads.

**Trace.** The tracer records, under `warnings._lock`, the `id()` and length of the list that each `catch_warnings.__enter__`, `catch_warnings.__exit__` and `resetwarnings()` acts on, plus the list each `_FuncWrapper` task was handed. In the 30 baseline attempts on the 4-core machine (366,000 thread tasks, 1,830 thread-backed `Parallel` calls), all 9,190 warnings fall into two paths:

1. *The handed list is emptied during the call* (2,390 warnings, 20 calls). `Parallel.__call__` hands the tasks `warnings._get_filters()`, which is the live `warnings.filters` object. A thread enters `catch_warnings`, another thread's exit makes the handed list global again, and the first thread's `resetwarnings()` then empties it. All 21 recorded resets have that order. The remaining tasks of the call warn.
2. *The next call captures an empty list* (6,800 warnings, 34 calls). After `fit`, the global list is a copy created by a worker thread's `__enter__` and left behind by an out-of-order `__exit__`. The `Parallel.__call__` in `predict_proba` captures it, and all 200 of its tasks warn.

A third outcome gives no warning. The leftover list is shortened but not empty, so line 143 does not fire and the tasks run with some filters missing (26 calls; 5,679 tasks ran with a shortened list).

**Copying the list is not enough.** With `Parallel.__call__` patched to hand tasks `list(warnings._get_filters())`, path 1 disappears (0 resets of a handed list in 30 attempts). However, 12 of 30 attempts still warn through path 2, and 20 of 30 still run tasks with a damaged list.

**A user's own filter does not hide it.** With `warnings.filterwarnings("ignore", message=...)` set before `fit`, 4,759 of 5,451 raised warnings were still shown, because the list in force when the warning is raised is usually one of the damaged ones.

**Deterministic reproducer.** [This script](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py) forces path 1 with two threads and four trivial tasks by ordering the two threads' entry and exit. It does not change the switch interval, and on every run it shows 11 filters becoming 0 and two warnings. It shows that this interleaving is sufficient to produce the warning. It does not show how often the interleaving happens.

**Limits.** Two machines, one scikit-learn version, one dataset and grid. The rates describe these setups only. The tracer adds work to each task and may change how often the race happens (21 of 30 traced against 5 of 10 untraced on the 4-core machine; the difference is not significant, Fisher p = 0.28). I did not examine whether `PYTHON_CONTEXT_AWARE_WARNINGS=1` has other effects, such as the cost on free-threaded builds mentioned earlier in this thread.

Report, tracer, analyzer and raw traces of the affected processes: [ROOT_CAUSE.md](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/RO