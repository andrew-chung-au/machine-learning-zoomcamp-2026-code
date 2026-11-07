# Findings: probable cause of the scikit-learn parallel warning

Status: **probable cause identified; not proven for the real workload.**
Supersedes the "cause remains unresolved" conclusion in `INVESTIGATION.md`
only to the extent stated below. Evidence levels are labelled throughout.

## Summary

The warning is most likely a **false positive** produced by scikit-learn
itself, triggered by a thread-safety problem between
`sklearn.utils.parallel._FuncWrapper` and `warnings.catch_warnings()` on
Python without context-aware warnings. It does not indicate that the user
code mis-pairs `delayed` and `Parallel`.

## 1. The warning condition (verified from source)

scikit-learn 1.9.1, `sklearn/utils/parallel.py`:

```python
141  config = getattr(self, "config", {})
142  warning_filters = getattr(self, "warning_filters", [])
143  if not config or not warning_filters:
144      warnings.warn("`sklearn.utils.parallel.delayed` should be used with ...")
```

The test is `not config or not warning_filters`. It fires when the captured
**warning-filter list is empty**, not only when `Parallel` was bypassed.
`Parallel.__call__` captures that list with `warnings._get_filters()` (or
`warnings.filters`) at the moment it is called.

## 2. What the real captures show (observed)

`real-probe-20261107T173304Z/capture-944567.jsonl`, 200 target-warning records:

| field | value |
| --- | --- |
| `captured_config_empty` | `false` (config **was** propagated) |
| `passed_warning_filters` | `[]` (filters empty) |
| `initial_warning_filters` (same PID, at startup) | 5 filters |
| stack | `threading` -> `multiprocessing/pool.py worker` -> `joblib BatchedCalls` -> `sklearn/utils/parallel.py:144 __call__` |

Interpretation: `sklearn.Parallel` **was** used (config present); the
warning fired only because the filter list was empty. The stack is a
**thread-pool worker**, i.e. joblib's threading backend, which is what
`RandomForestClassifier(n_jobs=-1)` uses internally.

Pattern across all logs in `07-midterm/` (observed):

| log | grid `n_jobs` | forest `n_jobs` | runs with warning |
| --- | --- | --- | --- |
| `reproduce_warning.log` | -1 | 1 | 0 of 3 |
| `reproduce_warning.log` | -1 | -1 | 2 of 3 (runs 4 and 6) |
| `reproduce_warning_traceback.log` | 1 | -1 | 0 of 5 |
| `minimal_reproducer_traceback.log` | -1 | -1 | 1 of 3 (run 3) |
| `reproduce_warning_stack.log` | -1 | -1 | 1 of 5 (run 5) |

The warning appears only when threads are used inside the workers, is
intermittent, and tends to appear after earlier runs. This is consistent with
the mechanism in section 3. It is a small sample and not a controlled test.

## 3. Proposed mechanism

With threads and no context-aware warnings, `warnings.filters` is one list for
the whole process, and `catch_warnings` is documented as not thread-safe. Each
sklearn task does `catch_warnings()` + `resetwarnings()` + re-add. If a second
thread enters between the first thread's reset and re-add, it snapshots the
half-reset list. When the threads exit out of order, the global list is
restored to that half-reset snapshot. The next `Parallel.__call__` in that
process captures the truncated or empty list and every task warns. In loky
worker processes nothing restores the list afterwards, so the damage persists
for later tasks that reuse the worker.

This also explains why `warnings.filterwarnings("ignore", ...)` in the
notebook did not help: the user's filter is part of the list that was lost, and
the worker re-applies the empty list inside each task.

## 4. Stdlib reproduction of the mechanism (observed, but a surrogate)

`thread_filter_race_probe.py` copies the wrapper's filter handling and runs it
in threads. It uses no scikit-learn, joblib or dataset. Python 3.14.4, 60
trials, switch interval shortened to 1e-6 s:

| setting | final filter count (started with 5) | next `Parallel` call would warn |
| --- | --- | --- |
| default | `{0: 19, 1: 10, 2: 12, 3: 11, 4: 8}` | 19 / 60 |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | `{5: 60}` | 0 / 60 |

With the default 5 ms switch interval: 0 of 20 trials corrupted. The
shortened interval is an **amplification** that shows the interleaving is
possible, not that it is common. It does **not** show the real
`GridSearchCV` run hits the same interleaving.

## 5. Upstream

scikit-learn issue **#35089** ("thread-based `Parallel` can corrupt the
process-wide warning filters", open, labels: Bug, free-threading, Needs
Investigation) describes the same mechanism and the same spurious warning.
Related: #32631 (closed, intermittent CI failure on free-threaded 3.13).

In that thread the reporter wrote that they could not trigger the race on GIL
Python with a realistic workload. The logs in this folder were captured on a
GIL build (Python 3.14.5, `Py_GIL_DISABLED = 0`) with a realistic workload, so
they are relevant new evidence for that issue.

## 6. What this does and does not establish

Established:
- The warning condition fires on an empty filter list (source).
- In a real run the config was present and the filter list was empty (capture).
- The failing stack is a threading-backend worker (capture).
- A stdlib model of the wrapper corrupts the global filter list under thread
  interleaving, and context-aware warnings prevent it (probe).

Not established:
- That the interleaving is what emptied the list in the real run. No capture
  has recorded the transition (see section 7).
- That `PYTHON_CONTEXT_AWARE_WARNINGS=1` fixes the real workload. The probe
  suggests it should; a real-workload comparison is still needed.
- Any frequency or rate at default scheduling.

## 7. Defect in the existing diagnostics

`capture.py` `TrackingFilters` replaces `warnings.filters` with a list
subclass. On Python 3.14 (non-context-aware) `catch_warnings.__enter__`
installs a plain `list` copy, and the tracker is silently gone after the first
task. Verified on 3.14.4: the type is `list` inside the block and `TrackingFilters`
is restored only on exit. Consequently "no filter-transition events" in
`real-filter-trace-20261107T174128Z` is **not** evidence that no transition
occurred. Replace it with a check at the points that matter (see next steps).

## Next steps

1. Run the real workload with an in-task check instead of a list subclass:
   in the traced `warnings.warn`, also record `len(warnings.filters)` per
   process at every inner-`Parallel` start, so the first drop is captured.
2. A/B on the real workload, same machine, many repeats:
   `PYTHON_CONTEXT_AWARE_WARNINGS=0` vs `1`, forest `n_jobs=-1`, grid
   `n_jobs=-1`. Report warning counts per run, not per call.
3. Add the real-workload evidence to scikit-learn #35089.
4. Revise the FAQ entry (separately, with owner approval). See below.

## Suggested FAQ revision

- Drop Step 1 (`filterwarnings`) as a remedy; explain it cannot work here.
- State the workaround as environment-specific, with the evidence level from
  section 6, and link #35089.
- Keep the note that `n_jobs=1` on the forest avoided the warning in the
  logged runs, as an observation.
