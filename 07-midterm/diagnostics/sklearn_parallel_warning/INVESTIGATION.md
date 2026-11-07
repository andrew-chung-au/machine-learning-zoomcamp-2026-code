# Investigation: scikit-learn parallel warning

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

## Evidence locations

- `real-probe-20261107T173304Z/`: historical real probe and per-process
  JSONL capture files.
- `real-filter-trace-20261107T174128Z/`: later real run with filter tracing.
- `filter-trace-test-20261107T174059Z/`: synthetic filter-tracing evidence.
- `test_capture.py` and `test_filter_tracking.py`: diagnostic regression tests.
- `README.md`: diagnostic suite operation, interpretation, and limitations.

## Change policy

Keep the original FAQ, its notebook, the dataset, and its attribution
unchanged. Make future investigation changes in the separate investigation
artefacts. Do not suppress the warning or patch installed library code merely
to obtain a clean run.
