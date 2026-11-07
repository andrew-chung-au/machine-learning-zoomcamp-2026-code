# Investigation report: scikit-learn parallel warning in the midterm FAQ entry

| | |
| --- | --- |
| Date | 2026-11-07 |
| Subject | FAQ entry on `GridSearchCV` parallel processing and the `sklearn.utils.parallel.delayed` warning |
| Trigger | Maintainer review of the FAQ pull request: the submitted example could not be verified, and a reproducer was requested |
| Status | Probable cause identified and supported by a controlled comparison; the mechanism itself has not been observed directly |
| Evidence labels | **source** (library code), **capture** (instrumented real runs), **probe** (standard-library surrogate), **A/B** (controlled real-workload comparison) |

## 1. Summary

The FAQ entry presented `PYTHON_CONTEXT_AWARE_WARNINGS=1` as the resolution
for a scikit-learn parallel-processing warning. It was written on the basis of
a single observation in a single environment, without a runnable reproducer.
It also led with a conventional warning-filter step whose result in the
author's own environment (ineffective) was not stated. The maintainer's
objection that the fix had not been established was correct.

The investigation that followed found that the warning is most likely a
**false positive raised by scikit-learn itself**. It is not caused by the
user's code mis-pairing `delayed` and `Parallel`. The warning appears to be
the visible symptom of a thread-safety problem: concurrent worker threads
corrupt the process-wide warning-filter list, and scikit-learn then reads an
empty list as "configuration was not propagated".

On the real workload (grid search `n_jobs=-1`, random forest `n_jobs=-1`),
15 paired attempts gave:

| `PYTHON_CONTEXT_AWARE_WARNINGS` | attempts with the warning |
| --- | --- |
| 0 | 8 of 15 (53%) |
| 1 | 0 of 15 (0%) |

Fisher exact test, two-sided: p = 0.0022. The result supports the proposed
explanation. It does not prove it, because the moment the filter list is
emptied has never been recorded. Two further observations are also not yet
explained (section 5).

The FAQ entry should be revised before it is relied on (section 8).

## 2. What went wrong

1. **The fix was stated as a result without evidence to support it.** The FAQ
   entry said the setting "resolved the warning". The basis was one
   environment and no repeated or controlled comparison. The warning is
   intermittent: in the later controlled comparison it appeared in about half
   of the attempts with the setting off, so a single clean run says very
   little.
2. **The example in the submission was not runnable.** It used names for the
   estimator, parameter grid and data that were never defined, so a reviewer
   could not reproduce the problem or test the proposed fix.
3. **The result of Step 1 in the author's environment was not reported.** The
   entry offered a targeted `warnings.filterwarnings(...)` first. This was a
   reasonable inclusion: targeted suppression is the usual first attempt in
   earlier FAQ entries, and the author's environment might have been unusual,
   so it was offered for readers whose situation differed. The entry framed it
   conditionally ("if the warning still appears, continue to Step 2"). The gap
   is that the filter had already been tried and had **not** worked in the
   author's environment, and the entry did not say so. Readers could not tell
   that the first step was known to fail in the one environment tested.
   The mechanism in section 4.3 explains the failure: when the captured filter
   list is empty, the user's filter is not re-applied in the worker. Whether
   the filter works in other environments, for example where the list is not
   emptied, was not tested.
4. **The scope of the claim exceeded the evidence.** The entry did note that
   the workaround was not guaranteed, but the headline and summary still read
   as a fix.
5. **The first diagnostic suite could not detect the event it was built to
   find.** See section 6.

## 3. Reproducer

Requested by the maintainer: estimator, parameter grid, exact command,
traceback, and whether the code uses `joblib` or scikit-learn's own
`Parallel` / `delayed`.

| Item | Value |
| --- | --- |
| Estimator | `RandomForestClassifier(random_state=1, n_jobs=-1)` |
| Parameter grid | `n_estimators=[200]`, `max_depth=[10, 20, None]`, `min_samples_leaf=[1]`, `max_features=["sqrt", 0.5]` |
| Search | `GridSearchCV(scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=1), n_jobs=-1)` |
| Data | `07-midterm/data/training-dataset.arff`: the UCI Phishing Websites dataset (CC BY 4.0, attribution in `07-midterm/data/README.md`), chosen for the author's own midterm project; training split of 8,844 rows and 30 features |
| Own use of `joblib` / `Parallel` / `delayed` | None. The code uses `GridSearchCV` and `RandomForestClassifier` only; `joblib` is imported solely to print its version |
| Environment | Python 3.14.5 (GIL build), scikit-learn 1.9.1, joblib 1.6.0, NumPy 2.5.4, SciPy 1.17.1 |
| Command (single run) | `uv run python 07-midterm/reproduce_sklearn_parallel_warning.py --grid-n-jobs -1 --forest-n-jobs -1 --repeat 3` |
| Command (controlled comparison) | `uv run python 07-midterm/diagnostics/sklearn_parallel_warning/ab_context_aware.py --pairs 15` |

The warning is intermittent. A single run may be clean; repeat the command.

Traceback of the emitting thread (from `capture-970401.jsonl`, innermost
frame last):

```text
threading.py, line 1044, in _bootstrap
threading.py, line 1082, in _bootstrap_inner
threading.py, line 1024, in run
multiprocessing/pool.py, line 125, in worker
joblib/_utils.py, line 109, in __call__
joblib/parallel.py, line 607, in __call__
sklearn/utils/parallel.py, line 144, in __call__
    warnings.warn(
```

Warning text: `` `sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel` to make it possible to propagate the
scikit-learn configuration of the current thread to the joblib workers. ``

## 4. Findings

### 4.1 The warning condition (source)

In scikit-learn 1.9.1, `sklearn/utils/parallel.py` lines 141-144:

```python
config = getattr(self, "config", {})
warning_filters = getattr(self, "warning_filters", [])
if not config or not warning_filters:
    warnings.warn("`sklearn.utils.parallel.delayed` should be used with ...")
```

The test fires when the captured warning-filter list is **empty**, in
addition to when `Parallel` was bypassed. The message therefore
misdescribes the situation whenever `Parallel` was used correctly and the
filter list happens to be empty.

### 4.2 What the real runs recorded (capture)

Across all 1,703 warning records in `ab-runs/20261107T193923Z/`:

| Field | Result |
| --- | --- |
| Configuration present (`captured_config_empty = false`) | 1,703 of 1,703 |
| Captured filter list length | 0 in 1,703 of 1,703 |
| Emitter | `sklearn/utils/parallel.py:144` in 1,703 of 1,703 |
| Stack | thread-pool worker in 1,703 of 1,703 |
| Filters at process startup (emitting processes) | 5 |

The same signature appears in the earlier records
(`real-probe-20261107T173304Z/capture-944567.jsonl`, and PID 970401 in
`runs/20261107T185047Z`). In every case scikit-learn's `Parallel` was used and
the configuration was propagated; the warning fired because the filter list
was empty.

Pattern across all logged runs: the warning appeared only when the forest used
`n_jobs=-1` **and** the grid search used worker processes. It did not appear
with a single-threaded forest in any logged run.

### 4.3 Proposed mechanism (hypothesis)

Without context-aware warnings, `warnings.filters` is one list shared by the
whole process, and `catch_warnings` is documented as not thread-safe. Each
scikit-learn task runs `catch_warnings()`, `resetwarnings()`, then re-adds the
captured filters. If a second thread enters between the first thread's reset
and re-add, it saves the half-reset list. When the threads leave out of order,
the global list is restored to that half-reset state. The next `Parallel`
call in the same process captures the truncated or empty list, and every task
then warns. In worker processes nothing restores the list afterwards, so the
damage can persist across later tasks.

This would also account for the failure of the Step 1 filter in the tested
environment: the user's filter is part of the list that was lost.

The same mechanism is described in the scikit-learn tracker as
[#35089](https://github.com/scikit-learn/scikit-learn/issues/35089)
("thread-based `Parallel` can corrupt the process-wide warning filters", open
when last checked; related: #32631, closed). The reporter there stated that
the race could not be triggered on a GIL build with a realistic workload. The
data in this folder come from a GIL build (Python 3.14.5) with a realistic
workload, and are therefore relevant to that issue.

### 4.4 Standard-library reproduction of the mechanism (probe)

`thread_filter_race_probe.py` copies the wrapper's filter handling and runs it
in threads, with no scikit-learn, joblib or dataset. Python 3.14.5, 60 trials,
switch interval shortened to 1e-6 s (`probe_default.log`,
`probe_context_aware.log`):

| Setting | Filters remaining (started with 5) | Next `Parallel` call would warn |
| --- | --- | --- |
| default | `{0: 16, 1: 13, 2: 11, 3: 8, 4: 12}` | 16 / 60 |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | `{5: 60}` | 0 / 60 |

A separate run on Python 3.14.4 gave the same pattern (19/60 and 0/60). At the
default 5 ms switch interval the probe produced no corruption in 20 trials.
The shortened interval is a deliberate amplification: it shows that the
interleaving is possible, not that the real run reaches it this way.

### 4.5 Controlled comparison on the real workload (A/B)

`ab_context_aware.py --pairs 15`, evidence in `ab-runs/20261107T193923Z/`.
Grid search `n_jobs=-1` and forest `n_jobs=-1`; a fresh process per attempt;
arms alternated with the order swapped each pair; `PYTHONWARNINGS` removed.

| `PYTHON_CONTEXT_AWARE_WARNINGS` | Attempts | Valid | With warning | Rate |
| --- | --- | --- | --- | --- |
| 0 | 15 | 15 | 8 | 53% |
| 1 | 15 | 15 | 0 | 0% |

Fisher exact p = 0.0022. No attempt was excluded and all 30 exited 0. Each
attempt's capture records report the intended `context_aware_warnings` value
in all 19 processes, so the setting did reach the workers.

Supporting runs:

- `runs/20261107T191042Z` (setting on, `--repeat 5`): 0 warnings in 20
  attempts, 0 of 5 for the relevant configuration. All 205 startup records
  report the setting as 1. Taken alone this is weak evidence: at a baseline of
  about one third, 0 of 5 has roughly a 10% chance of occurring anyway.
- `runs/20261107T185047Z` (setting off): the relevant configuration warned in
  1 of 3 attempts; the other three configurations did not warn in 9 attempts.

## 5. Assessment of confidence

**Established**

- The warning fires on an empty filter list (source).
- In every captured record the configuration was present and the captured
  filter list was empty, in a threading-backend worker (capture).
- On this machine and stack, enabling context-aware warnings removed the
  warning in 15 of 15 attempts, against 8 of 15 with it disabled (A/B).
- A standard-library model of the wrapper corrupts the global filter list
  under thread interleaving, and context-aware warnings prevent this (probe).

**Not established**

- That the thread race is what emptied the list in the real run. The change
  from 5 filters to an empty list has not been recorded. The A/B shows that the
  setting removes the symptom; since it changes how `catch_warnings` stores
  state for every thread, the result is consistent with the race but does not
  isolate it from other effects.
- **Open observation 1.** The process-wide filter list was not always empty at
  warning time: 0 entries in 1,420 records, and 1, 2, 12, 13 or 14 in the
  other 283. The captured list was empty every time. Something continues to
  change the global list after the empty capture. This has not been
  investigated.
- **Open observation 2.** Warning counts per process were not multiples of 200
  (2, 27, 54, 75, 122, 171, 200 three times, 252, 400). With 200 trees, 200 per
  affected `Parallel` call would be expected. Partial counts mean either that
  not every call in a process captured an empty list, or that records were
  missed. Neither explanation has been tested.
- Behaviour outside one machine, Python 3.14.5, scikit-learn 1.9.1, joblib
  1.6.0, this dataset and this grid. Thirty attempts is a small sample, and the
  53% rate applies to this setup only.

## 6. Errors in the investigation's own tooling

1. **The filter tracker was blind to the event it was written to detect.**
   `TrackingFilters` in `capture.py` replaces `warnings.filters` with a list
   subclass. On Python 3.14 without context-aware warnings,
   `catch_warnings.__enter__` installs a plain `list` copy, so the tracker is
   gone after the first task. This was verified on Python 3.14.4.
2. **A conclusion drawn from that tracker was unsound.** The earlier note that
   `real-filter-trace-20261107T174128Z` showed "no filter-transition events" was
   treated as meaningful. It is not evidence that no transition occurred.
   `INVESTIGATION.md` retains this wording and should be read with that
   correction.
3. **An early reading overstated the evidence.** A filter length of 0 at
   warning time in one record was read as confirming that the global list was
   empty. Section 5 shows that this holds for most records but not all.
4. **The first matrix run with the setting on was nearly mistaken for a
   result.** Zero warnings in 5 attempts is likely by chance (about 10%). The
   paired comparison was added for this reason.

## 7. Impact

- The published FAQ guidance may mislead readers: it implies a general fix,
  and it does not say that its first step failed in the tested environment.
- No effect on model results was found. Best score and parameters were
  identical across runs (best ROC AUC about 0.99607); the warning is a
  configuration-propagation message and does not indicate a wrong calculation.

## 8. Corrective actions

| # | Action | Owner | Status |
| --- | --- | --- | --- |
| 1 | Publish a runnable reproducer with the estimator, grid, command and traceback requested by the maintainer (section 3) | Author | Done in this report |
| 2 | Run a controlled comparison rather than rely on a single observation | Author | Done (section 4.5) |
| 3 | Revise the FAQ entry (below) | Author | Done (`07-midterm/07-faq_pr.md`) |
| 4 | Record the transition from 5 filters to an empty list directly: log `len(warnings._get_filters())` with timestamp, process ID and thread ID at each inner `Parallel` start and in the warning hook. Do not rely on a list subclass | Author | **Pending** |
| 5 | Investigate open observations 1 and 2 using those records | Author | **Pending** |
| 6 | Add the real-workload evidence to scikit-learn #35089 | Author | **Pending** |
| 7 | Annotate `INVESTIGATION.md` with the correction in section 6 | Author | Done |

**Proposed FAQ revision**

- Keep the `filterwarnings` step as the conventional first attempt, but state
  that it did not work in the tested environment, explain the likely reason
  (section 4.3), and say that its effect elsewhere is untested.
- Present `PYTHON_CONTEXT_AWARE_WARNINGS=1` as a workaround that removed the
  warning in 15 of 15 attempts on one stack (against 8 of 15 without it), not
  as a general fix, and link #35089.
- Replace "resolved the warning" with wording that matches the evidence levels
  in section 5.
- Keep the observation that a single-threaded forest (`n_jobs=1`) avoided the
  warning in every logged run.
- Use a self-contained example with every name defined.

## 9. Evidence index

All paths are relative to `07-midterm/diagnostics/sklearn_parallel_warning/`
unless noted.

| Item | Location |
| --- | --- |
| Controlled comparison | `ab-runs/20261107T193923Z/` (`results.json`, `summary.jsonl`, per-attempt `manifest.json`, `stdout_stderr.log`, `capture/*.jsonl`) |
| Matrix runs | `runs/20261107T185047Z/` (setting off), `runs/20261107T191042Z/` (setting on) |
| Earlier real probe | `real-probe-20261107T173304Z/` |
| Standard-library probe | `thread_filter_race_probe.py`, `probe_default.log`, `probe_context_aware.log` |
| Comparison script | `ab_context_aware.py` |
| Reproducer | `../../reproduce_sklearn_parallel_warning.py` |
| Earlier notes (see section 6 correction) | `INVESTIGATION.md` |
| Upstream issue | https://github.com/scikit-learn/scikit-learn/issues/35089 |

Interpretation notes: the unit in the comparison is the attempt, not the
warning count, since one attempt can log hundreds of warnings from a single
event. A finite set of attempts cannot prove that a configuration never emits
the warning.
