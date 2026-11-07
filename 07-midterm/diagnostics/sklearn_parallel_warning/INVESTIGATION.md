# Investigation: scikit-learn parallel warning

> **Update 2026-11-07: superseded in part. Read `FINDINGS.md` first.**
>
> This file records the first stage of the investigation and is kept
> unchanged below so the history stays visible. Later work changed three of
> its statements:
>
> 1. **"The underlying cause remains unresolved" is no longer the current
>    position.** A probable cause is now identified (thread-unsafe handling
>    of `warnings.filters` in scikit-learn's worker wrapper) and supported by a
>    controlled comparison: 8 of 15 attempts warned with
>    `PYTHON_CONTEXT_AWARE_WARNINGS=0`, against 0 of 15 with it set to 1
>    (p = 0.0022). The mechanism itself has still not been observed directly.
>    See `FINDINGS.md`, sections 1, 4.5 and 5.
> 2. **The "later real run did not capture the warning" and "no
>    filter-transition events" findings (Findings 3 and 4 below) must not be
>    read as evidence that no transition occurred.** The filter tracker in
>    `capture.py` replaces `warnings.filters` with a list subclass, and
>    `catch_warnings()` swaps in a plain `list` copy on Python 3.14, so the
>    tracker is gone after the first task. It could not have detected the
>    event. The absence of transition events carries no information.
>    See `FINDINGS.md`, section 6. The tracker still needs to be replaced
>    (`FINDINGS.md`, section 8, item 4).
> 3. **The statement that the original FAQ "must remain unchanged" has been
>    lifted for `07-faq_pr.md`.** That entry has been revised to match the
>    evidence. The notebook, dataset and attribution remain unchanged.
>
> The per-process counts, the 200-warning figure and the other numbers below
> are unaffected.

## Purpose and scope

This investigation examines the warning:

`sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel`

The warning arose while testing the existing midterm FAQ and its notebook.
The original FAQ, notebook, dataset, and attribution are source material for
the investigation and must remain unchanged.

The investigation uses a separate reproducer, diagnostic instrumentation,
captured logs, and regression tests. It does not patch installed scikit-learn
source code.

## Findings

### 1. The warning has been observed in a real run

The historical probe in `real-probe-20261107T173304Z` captured 200 matching
warning calls in one process (PID 944567).

The recorded warning emitter was in scikit-learn's
`sklearn/utils/parallel.py`, in `_FuncWrapper.__call__`, at line 144 in the
installed version examined during that run.

This establishes that the instrumented reproducer observed the warning. The
200 calls are not 200 independent experiments.

### 2. Warning-filter state differed between startup and warning capture

The captured process startup record reported five warning filters. At the
captured warning calls, the passed warning-filter state was empty, and the
recorded current filter length was zero.

This is evidence of a difference between the recorded startup and warning
states. It does not establish when the state changed, what changed it, or
whether that change caused the warning.

### 3. A later real run did not capture the warning

The run in `real-filter-trace-20261107T174128Z` completed its model workload
with the same reported best score, approximately 0.99607, but captured no
matching target warnings and no filter-transition events.

This is a no-warning observation, not evidence that the warning has been
fixed. The absence of a captured warning also does not prove that every
relevant process and warning path was observed.

> **Correction (2026-11-07):** the "no filter-transition events" part of this
> finding is not meaningful, because the tracker could not see transitions
> (see the update note at the top of this file). The no-warning observation
> itself stands, and fits the later finding that the warning is intermittent
> (about half of attempts with context-aware warnings off).

### 4. The diagnostic instrumentation passed its regression tests

The synthetic joblib capture test, filter-tracking regression tests, and
Python syntax checks passed: 3/3 checks.

These tests validate aspects of the diagnostic machinery. They do not
establish the root cause of the warning in the real workload.

## Limitations

- The observed warning behaviour has not been shown to occur on every run.
- A captured warning count is not a count of independent reproductions.
- The capture mechanism instruments Python's `warnings.warn`; it is not a
  general detector for every possible warning-emission path.
- A capture-start record does not prove that every worker was instrumented.
- Filter tracking detects selected transitions and replacement conditions;
  it does not record every possible filter change.
- The evidence does not establish why the warning-filter state differed or
  prove that the filter state caused the warning.
- The results apply to the environment and package versions recorded in the
  relevant run evidence. They should not be generalized to all environments.

## Current conclusion

The warning has been observed in the instrumented real workload. One probe
recorded empty warning-filter state at warning capture despite a non-empty
startup state. A later run completed without capturing the warning, and the
synthetic diagnostic tests passed.

The underlying cause remains unresolved. The intermittent no-warning result
must not be treated as a fix, and the filter-state observation must not be
presented as a confirmed causal explanation.

> **Update (2026-11-07):** a probable cause has since been identified and
> tested; see `FINDINGS.md`. The caution above about not presenting the
> filter-state observation as confirmed causation still applies to the
> mechanism, which remains unobserved.

## Evidence locations

- `real-probe-20261107T173304Z/`: historical real probe and per-process
  JSONL capture files.
- `real-filter-trace-20261107T174128Z/`: later real run with filter tracing.
- `filter-trace-test-20261107T174059Z/`: synthetic filter-tracing evidence.
- `test_capture.py` and `test_filter_tracking.py`: diagnostic regression tests.
- `README.md`: diagnostic suite operation, interpretation, and limitations.

## Change policy

Keep the notebook, the dataset, and its attribution unchanged. The FAQ entry
(`07-faq_pr.md`) was revised on 2026-11-07 after maintainer review, to match
the evidence in `FINDINGS.md`; its earlier text is in git history. Make future
investigation changes in the separate investigation artefacts. Do not suppress the warning or patch installed library code merely
to obtain a clean run.
