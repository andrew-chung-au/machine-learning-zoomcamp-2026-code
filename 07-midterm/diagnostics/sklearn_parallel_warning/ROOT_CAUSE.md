# Root cause: the scikit-learn "`delayed` should be used with `Parallel`" warning

| | |
| --- | --- |
| Date | 2026-11-11 |
| Follows | [`FINDINGS.md`](FINDINGS.md) (2026-11-07), which identified a probable cause but had not observed it |
| Status | Cause observed directly and reproduced deterministically. Two changes that remove it are supported by direct observation on two machines. The defect is in scikit-learn and is not fixed upstream |
| Scope of evidence | Two machines: a 4-core machine with Python 3.14.6 (sections 4 to 6) and the original 16-core machine with Python 3.14.5 (sections 5.1 and 6). GIL builds, scikit-learn 1.9.1, joblib 1.6.0, one dataset and grid. Limits are in section 8 |

## 1. Conclusion

The warning is a false alarm raised by scikit-learn. The code that triggers
it pairs `delayed` and `Parallel` correctly, and scikit-learn's configuration
reaches the workers in every recorded warning.

**Cause.** In every parallel task scikit-learn 1.9.1 swaps Python's
warning-filter list (`warnings.catch_warnings()`), empties it
(`warnings.resetwarnings()`) and re-adds the caller's filters. On Python
without context-aware warnings that list is one process-wide object, and these
operations are not safe when several threads run them at once.
`RandomForestClassifier(n_jobs=-1)` runs its tasks in threads. The threads
overwrite each other's swaps, and a filter list ends up empty. scikit-learn
then reads "the filter list I was given is empty" as "`delayed` was used
without `Parallel`" and warns.

**What removes it** (real workload, 30 attempts each, section 5):

| Change | Attempts with the warning | Attempts with a damaged filter list |
| --- | --- | --- |
| None (grid search and forest both parallel) | 21 of 30 | 26 of 30 |
| Forest single-threaded, grid search still parallel | 0 of 30 | 0 of 30 |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 30 | 0 of 15 (measured in the second run only) |

**What does not remove it:**

| Change | Result |
| --- | --- |
| The user's own `warnings.filterwarnings("ignore", ...)` | 12 of 15 attempts still showed the warning; 4,759 of 5,451 raised warnings were shown |
| Handing tasks a copy of the filter list (a one-line change to scikit-learn, tested as a patch) | 12 of 30 attempts still warned |

The same comparison on the original 16-core machine, with plain `n_jobs=-1`,
gave 13 of 15, 0 of 15 and 0 of 15 (section 5.1).

The first two rows of the first table do not fix scikit-learn. They avoid the
condition the defect needs (threads sharing one filter list). The defect
itself is tracked upstream as
[scikit-learn #35089](https://github.com/scikit-learn/scikit-learn/issues/35089),
open when checked on 2026-11-11; the code on scikit-learn's `main` branch was
identical to 1.9.1 on that date.

## 2. The defect

`sklearn/utils/parallel.py`, scikit-learn 1.9.1.

**Capture** (`Parallel.__call__`, lines 78-81):

```python
filters_func = getattr(warnings, "_get_filters", None)
warning_filters = (
    filters_func() if filters_func is not None else warnings.filters
)
```

Without context-aware warnings, `warnings._get_filters()` returns
`warnings.filters` itself. The tasks are handed the live list, not a copy.

**Check and replay** (`_FuncWrapper.__call__`, lines 141-182, run by every task):

```python
config = getattr(self, "config", {})
warning_filters = getattr(self, "warning_filters", [])
if not config or not warning_filters:        # line 143
    warnings.warn("`sklearn.utils.parallel.delayed` should be used with ...")

with config_context(**config), warnings.catch_warnings():   # line 154
    warnings.resetwarnings()                                 # line 156
    for filter_args in warning_filters:                      # line 158
        ...  # re-add each filter with append=True
    return self.function(*args, **kwargs)
```

**What the standard library does** (Python 3.14, `_py_warnings.py`, without
context-aware warnings):

- `catch_warnings.__enter__` saves the list that is currently global and makes
  a copy of it the global list.
- `catch_warnings.__exit__` makes the saved list global again.
- `resetwarnings()` empties whichever list is global at that moment.

The [`warnings` documentation](https://docs.python.org/3/library/warnings.html)
states that `catch_warnings` is not thread-safe unless context-aware warnings
are enabled.

**Three consequences when tasks run in threads.** With two or more threads the
saves and restores interleave. Lists L1, L2, ... are successive copies:

1. *The handed list is emptied during the call.*

   ```text
   thread A enters   saved = L1 (the list handed to the tasks)   global = L2
   thread B enters   saved = L2                                  global = L3
   thread A exits                                                global = L1
   thread B calls resetwarnings()   ->   empties L1
   ```

   Every task of that call that starts afterwards finds its handed list empty
   and warns.

2. *The next call captures an empty list.* A thread that enters while another
   thread's copy has just been emptied saves or copies an empty list. Exits
   happen in a different order from entries, so when the call ends the global
   list can be one of these leftovers instead of the original. The next
   `Parallel.__call__` (for example `predict_proba` during scoring, straight
   after `fit`) captures it, and all of its tasks warn.

3. *The same, but the leftover list is not completely empty.* No warning is
   raised, because line 143 only tests for an empty list, but the tasks run
   with some of the caller's filters missing.

The check on line 143 cannot tell "`Parallel` was not used" from "the filter
list is empty", so cases 1 and 2 produce a message that describes something
that did not happen.

## 3. Deterministic reproducer

[`root_cause/deterministic_reproducer.py`](root_cause/deterministic_reproducer.py)
runs scikit-learn's own `Parallel` and `delayed`, correctly paired, on the
threading backend with two threads and four trivial tasks. It fixes the order
in which the two threads enter and leave `catch_warnings` (the interleaving of
consequence 1). It does not change what the threads do, does not shorten the
interpreter's switch interval, and needs no dataset or estimator.

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py
```

Output (Python 3.14.6, scikit-learn 1.9.1, joblib 1.6.0; the same on every run):

```text
Before: 11 warning filters; the list handed to the tasks is the live warnings.filters object: True
After:  0 warning filters in the list handed to the tasks; results=[1, 2, 3, 4]
Target warnings emitted: 2
  {'thread': 'Thread-1 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
  {'thread': 'Thread-2 (worker)', 'config_present': True, 'stored_filters': 0, 'where': 'sklearn/utils/parallel.py:144'}
```

Tasks 3 and 4 warn although `delayed` and `Parallel` are paired and the
configuration is present.

| Mode | Warnings | Handed list after the call |
| --- | --- | --- |
| default | 2 | 0 filters (was 11) |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 | 11 filters |
| `--snapshot` (tasks get a copy; test patch) | 0 | 11 filters |

This shows the interleaving is sufficient. Section 4 shows it is what happens
in the real grid search.

## 4. Direct observation in the real grid search

### 4.1 Method

The workload is the reproducer already described in `FINDINGS.md` section 3
(`RandomForestClassifier` inside `GridSearchCV`, UCI Phishing Websites data).
A tracer
([`root_cause/trace/filter_identity_trace.py`](root_cause/trace/filter_identity_trace.py))
is loaded in the main process and in every joblib worker. It records, for
every `catch_warnings` entry, every `catch_warnings` exit and every
`resetwarnings()`, **which list object** the operation acted on (`id()`), its
length, the thread, and the time. For a scikit-learn task it also records the
list the task was handed. It does not depend on the list object surviving, so
it does not have the blind spot of the earlier tracker (`FINDINGS.md`
section 6).

[`root_cause/analyze_trace.py`](root_cause/analyze_trace.py) then looks for the
event behind every warning and checks the order of events around it.

Layout. The experiments ran on a 4-core machine. The original reports came
from a machine where `n_jobs=-1` gave 16 worker processes (19 traced processes
per attempt). To reproduce that layout the reproducer was given explicit
values: `--grid-n-jobs 16 --forest-n-jobs 16`, which also gives 19 processes
per attempt. Section 6 reports what happens with this machine's own
`n_jobs=-1`.

### 4.2 Result

Baseline arm of both comparison runs together
(`root_cause/runs/20261111T213534Z` and `root_cause/runs/20261111T215036Z`,
30 attempts, 570 processes, 366,000 tasks run in worker threads, 1,830 inner
`Parallel` calls):

| Measure | Value |
| --- | --- |
| Target warnings raised | 9,190, in 42 processes |
| ... with scikit-learn's configuration present | 9,190 of 9,190 |
| ... with an empty handed filter list | 9,190 of 9,190 |
| ... emitted from `sklearn/utils/parallel.py:144` in a worker thread | 9,190 of 9,190 |
| Warnings traced to a recorded cause | **9,190 of 9,190** |
| Cause 1: handed list emptied during the call | 2,390 warnings in 20 calls |
| Cause 2: next call captured an already-empty list | 6,800 warnings in 34 calls |
| Cause 3: next call captured a shortened list (no warning) | 26 calls |
| Tasks that ran with a shortened, non-empty filter list and no warning | 5,679 |

Checks on the recorded order of events:

- **Cause 1.** 21 resets emptied a handed list while it still had entries. In
  21 of 21 the resetting thread had entered `catch_warnings` earlier (saving a
  different list), another thread's exit then made the handed list global
  again, and only then did the first thread call `resetwarnings()`. That is
  the interleaving in section 2. The time between that thread's entry and its
  reset ranged from 1.6 ms to 420 ms (median 26 ms): the thread was paused
  between two adjacent statements while others ran.
- **Causes 2 and 3.** In 60 of 60 calls that captured an empty or shortened
  list, that list had been created by a worker thread's `catch_warnings` entry
  and left as the global list by a worker thread's `catch_warnings` exit in
  the previous call.
- No warning came from an outer (process-level) grid-search task, and none was
  left unexplained.

The damage does not accumulate across grid-search tasks. Each outer task runs
inside its own `catch_warnings` block in the worker's main thread, and its
exit restores the process's original list after the threads have finished.
That is why warnings come in bursts of at most 400 per outer task (one `fit`
call and one `predict_proba` call, 200 trees each).

### 4.3 Worked example

[`root_cause/example-trace/trace-24744.jsonl.gz`](root_cause/example-trace/)
is the raw trace of one worker process (attempt 46 of the first run, 382
warnings: 182 by cause 1, then 200 by cause 2).

```bash
uv run python 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/analyze_trace.py 07-midterm/diagnostics/sklearn_parallel_warning/root_cause/example-trace --show 1
```

Extract (`STORED` is the list `Parallel.__call__` handed to the tasks; times
are relative to the decisive reset):

```text
 -26012.6 us  T2   enter  catch_warnings: saved list@00480 (len 12), global := list@00672  [task handed STORED, len 12]
 -26005.4 us  T3   enter  catch_warnings: saved list@00672 (len 12), global := list@00416  [task handed STORED, len 12]
      ...          (more threads enter, each saving the previous thread's copy)
 -25882.5 us  T15  exit   catch_warnings: global := list@25952 (len 12)
 -22964.8 us  T15  exit   catch_warnings: global := STORED (len 12)
      0.0 us  T3   resetwarnings(): empties STORED (len 12 -> 0)  <== empties the stored list
    459.7 us  T4   resetwarnings(): empties STORED (len 0 -> 0)
```

Thread T3 entered, was paused for 26 ms before its `resetwarnings()`, and in
the meantime thread T15's exit made the stored list global. Sixteen tasks had
entered before the reset and two more had already passed the check; the
remaining 182 tasks of that call warned.

## 5. Fixes compared on the real workload

[`root_cause/compare_fixes.py`](root_cause/compare_fixes.py): one fresh process
tree per attempt, 15 attempts per arm per run, arm order rotated every round,
tracer on, `PYTHONWARNINGS` removed, 16 x 16 layout. The unit is the attempt.
"Damaged" means at least one task ran with an emptied or shortened filter
list, with or without a warning.

| Arm | Run 1 warned | Run 1 damaged | Run 2 warned | Run 2 damaged | Total warned |
| --- | --- | --- | --- | --- | --- |
| Baseline | 9 of 15 | 13 of 15 | 12 of 15 | 13 of 15 | **21 of 30** |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 15 | not measured | 0 of 15 | 0 of 15 | **0 of 30** |
| Forest `n_jobs=1` (grid search still 16 processes) | 0 of 15 | 0 of 15 | 0 of 15 | 0 of 15 | **0 of 30** |
| Snapshot test patch | 6 of 15 | 12 of 15 | 6 of 15 | 8 of 15 | **12 of 30** |
| User's `filterwarnings("ignore", ...)` | not run | not run | 12 of 15 | 14 of 15 | 12 of 15 |

Fisher exact test, two-sided, against baseline:

| Comparison | p |
| --- | --- |
| Warned: context-aware 0 of 30 against 21 of 30 | 3.6e-9 |
| Warned: single-threaded forest 0 of 30 against 21 of 30 | 3.6e-9 |
| Damaged: single-threaded forest 0 of 30 against 26 of 30 | 7.8e-13 |
| Damaged: context-aware 0 of 15 against 13 of 15 (run 2) | 1.8e-6 |
| Warned: snapshot patch 12 of 30 against 21 of 30 | 0.037 |

Notes on each arm:

- **Context-aware warnings.** Filters are then stored per context, so a
  thread's `catch_warnings` and `resetwarnings()` act on its own list. In run
  2 all 183,000 thread tasks were traced and every one was handed a full list.
  (Run 1 used an earlier tracer that recorded only the warning itself in this
  mode.)
- **Single-threaded forest.** No task runs in a worker thread (0 traced), so
  there is nothing to interleave. The grid search still uses all its worker
  processes.
- **Snapshot patch.** Copying the list in `Parallel.__call__` removes cause 1
  (0 decisive resets in 30 attempts) but not causes 2 and 3: the copy is taken
  from a global list that is already damaged. All 3,800 warnings in this arm
  were cause 2. A complete upstream fix needs more than the copy. This arm
  exists to inform the upstream issue; it is not something a user can apply.
- **User filter.** The reproducer's `--user-filter` option installs the
  targeted `ignore` filter in the main process before fitting, as the FAQ's
  earlier "Step 1" did. scikit-learn passes it to the workers correctly, but
  the warning is raised while the list in force is one of the damaged ones, so
  the filter is usually not there to apply. 4,759 of the 5,451 warnings raised
  were still printed (87%). This matches the baseline arm, where the list in
  force at the moment of the warning had two or fewer entries for 8,361 of
  9,190 warnings (91%).

### 5.1 Replication on the original machine

`root_cause/runs/20261112T032306Z`: the machine of the original reports
(16 logical CPUs, Python 3.14.5), plain `n_jobs=-1` for both levels with no
explicit layout, 15 attempts per arm, tracer on, 19 processes per attempt.

| Arm | Warned | Damaged | Thread tasks traced |
| --- | --- | --- | --- |
| Baseline | 13 of 15 | 15 of 15 | 183,000 |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 15 | 0 of 15 | 183,000 |
| Forest `n_jobs=1` | 0 of 15 | 0 of 15 | 0 |
| Snapshot test patch | 10 of 15 | 15 of 15 | 183,000 |

Fisher exact test against baseline: warned p = 1.8e-6 for each fix, damaged
p = 1.3e-8 for each fix.

Baseline detail: 3,500 warnings, all with configuration present and an empty
handed list, and 3,500 of 3,500 traced to a recorded cause (500 by cause 1 in
15 calls, 3,000 by cause 2 in 15 calls). All 15 decisive resets have the
order of events in section 2, and all 43 captures of an empty or shortened
list were lists left global by a worker thread's exit. 6,703 tasks ran with a
shortened list and no warning. Warnings per process: 1, 13, 19, 23, 33, 80,
101, 141, 200 (12 processes), 289, 400. The best score and best parameters
were identical in all 60 attempts.

Median wall time with the tracer on: baseline 11.1 s, context-aware 10.8 s,
single-threaded forest 10.5 s.

Over both machines the baseline warned in 34 of 45 attempts and each fix in
0 of 45 (first 16-core run only).

#### User filter on the 16-core machine

`root_cause/runs/20261112T035229Z`: 15 attempts per arm, tracer on, plain
`n_jobs=-1`. This run repeated the baseline alongside the user-filter arm.

| Arm | Warned | Damaged |
| --- | --- | --- |
| Baseline | 13 of 15 | 15 of 15 |
| User's `filterwarnings("ignore", ...)` | 11 of 15 | 14 of 15 |

Fisher exact test against baseline: warned p = 0.65, damaged p = 1.0. As on the
4-core machine, the user's filter did not stop the warning.

#### Untraced runs on the 16-core machine

`root_cause/runs/20261112T040023Z`: 10 attempts per arm, tracer off, plain
`n_jobs=-1`.

| Arm | Warned | Median wall time |
| --- | --- | --- |
| Baseline | 3 of 10 | 10.6 s |
| `PYTHON_CONTEXT_AWARE_WARNINGS=1` | 0 of 10 | 11.0 s |
| Forest `n_jobs=1` | 0 of 10 | 9.9 s |

Fisher exact test against baseline: warned p = 0.21 for each fix. With 10
attempts per arm this run is too small to be significant on its own.

## 6. Controls

All under `root_cause/controls/`.

| Control | Layout | Tracer | Result |
| --- | --- | --- | --- |
| `20261111T220745Z` | 16 x 16 | **off** | Baseline warned in 5 of 10; context-aware 0 of 10; single-threaded forest 0 of 10. The warning does not depend on the tracer |
| `20261111T221422Z` | grid search sequential, forest 16 threads | on | 0 of 8 warned, 0 of 8 damaged (97,600 thread tasks) |
| `20261111T221702Z` | this machine's `n_jobs=-1` (4 x 4) | on | Baseline 0 of 10 warned, 0 of 10 damaged; single-threaded forest 0 of 10 |
| `20261111T222016Z` | this machine's `n_jobs=-1` (4 x 4) | off | Timing only (below); no warning in 5 baseline attempts (5 single-threaded-forest attempts also clean) |

How often the race happens depends strongly on the layout. With 16 processes
each running 16 threads it damaged a filter list in 26 of 30 attempts. With
one process and 16 threads, or 4 processes with 4 threads each, it was not
seen in 18 traced attempts. The mechanism does not need nesting (section 3
reproduces it in one process), so these zeros mean "not observed in this many
attempts", not "cannot happen".

Wall time, tracer off, median per attempt (a full grid search of 30 fits plus
the refit):

| Layout | Both parallel | Forest single-threaded | Attempts per arm |
| --- | --- | --- | --- |
| 16 x 16 (on 4 cores) | 13.1 s | 12.7 s | 10 |
| `n_jobs=-1` (4 x 4, on 4 cores) | 9.1 s | 8.7 s | 5 |

On this machine the single-threaded forest was not slower. The grid search
already runs one fit per core; adding threads inside each fit gives more
threads than cores.

## 7. What this explains

- **Why the message is wrong.** `Parallel` was used and the configuration was
  propagated in every recorded warning. Line 143 fires on an empty filter
  list.
- **Open observation 1 in `FINDINGS.md`** (the process-wide list was not
  always empty at warning time). The check reads the list the task was
  *handed*; the list that is *global* at that instant can be any thread's
  copy. Here it had 0, 1, 2, 3, 12, 13 or 14 entries. The original records
  show 0, 1, 2, 12, 13 and 14.
- **Open observation 2** (warning counts per process were not multiples of
  200). Cause 1 starts part-way through a call, so it gives 200 minus the
  tasks that had already passed the check; cause 2 gives exactly 200; a
  process can have both. Counts here: 2, 93, 115, 127, 136, 141, 143, 151,
  166, 200 (24 processes), 308, 312, 341, 347, 364, 368, 372, 382, 522. The
  original counts (2, 27, 54, 75, 122, 171, 200 three times, 252, 400) have
  the same shape. The original records hold no list identities, so they
  cannot be checked one by one.
- **Why a `filterwarnings` filter did not work** (section 5, user filter).
- **Why a single clean run means little.** About one attempt in three was
  clean in the layout that triggers it.
- **Upstream.** The reporter of #35089 described the same interleavings and
  wrote that they could not trigger the race on a GIL build with a realistic
  workload. The runs here are a GIL build, an ordinary `GridSearchCV` over a
  random forest, and the default switch interval.

## 8. Limits and open points

- Two machines, one scikit-learn version, one dataset and grid. The rates
  describe these setups only.
- On the 4-core machine the 16 x 16 layout was imposed to match the process
  and thread counts of the original reports. On the original 16-core machine
  the same result came from plain `n_jobs=-1` (section 5.1). The controls in
  section 6 were done on the 4-core machine only. Untraced runs were done on
  both machines (section 6 for the 4-core machine, section 5.1 for the 16-core
  machine).
- Why the resetting thread is paused between entering `catch_warnings` and
  calling `resetwarnings()` was not established. The pauses (1.6 to 420 ms)
  and the dependence on layout suggest the operating system suspending
  threads when there are far more threads than cores; contention on the
  `warnings` module lock is another candidate. Neither was tested.
- The tracer adds work to each task and holds `warnings._lock` around each
  recorded operation. It can change how often the race happens (baseline 21
  of 30 traced, 5 of 10 untraced; the difference is not significant, Fisher
  p = 0.28). It cannot change which list an operation acts on.
- The context-aware arm removes the race by design, and no damage was recorded
  in 15 traced attempts. Whether it has other effects on scikit-learn was not
  examined. Upstream notes a possible cost on free-threaded builds, where each
  task takes the `warnings` lock many times.
- **Side observation, not explained.** In 7 of 203 attempts the best
  cross-validated ROC AUC on the 4-core machine was 0.9960724230 and not 0.9960725525, a difference
  in the seventh decimal place. All 7 used a threaded forest in the 16 x 16
  layout (7 of 148 attempts with a threaded forest, 0 of 55 with a
  single-threaded one). Two of the 7 had no warning, and one of those was a
  context-aware attempt with no filter damage recorded, so the warning is not
  the cause. The best parameters were identical in all 203 attempts. A likely
  reason is the order in which threads add up per-tree predictions; this was
  not investigated.
- No fix for scikit-learn itself is proposed or tested here beyond showing
  that copying the list is not enough.

## 9. Corrections to earlier documents

- `FINDINGS.md` section 4.3 described cause 2 (a truncated global list
  captured by the next call) as a hypothesis. It is now observed, and it
  accounts for about three quarters of the warnings here. Cause 1 (the handed
  list is the live list and is emptied during the call) was not in that
  report.
- `FINDINGS.md` section 4.4: `thread_filter_race_probe.py` copies the captured
  list (`list(...)`), so it cannot show cause 1. Its results remain valid for
  cause 2.
- `FINDINGS.md` section 8, actions 4 and 5 (record the transition; investigate
  the two open observations) are done here. Action 6 (add the evidence to
  #35089) has a draft in
  [`root_cause/upstream_comment_draft.md`](root_cause/upstream_comment_draft.md)
  and has not been posted.
- The FAQ entry (`07-midterm/07-faq_pr.md`) presented context-aware warnings
  as the main workaround and the single-threaded forest as an untested
  observation. Both are now tested. The entry now leads with the
  single-threaded forest, which needs no particular Python version and no
  restart.

## 10. Evidence index

Paths are relative to `07-midterm/diagnostics/sklearn_parallel_warning/`.

| Item | Location |
| --- | --- |
| Deterministic reproducer | `root_cause/deterministic_reproducer.py` |
| Tracer | `root_cause/trace/` |
| Analyzer | `root_cause/analyze_trace.py` |
| Comparison runner, aggregator | `root_cause/compare_fixes.py`, `root_cause/summarize_run.py` |
| Comparison run 1 (4 arms) | `root_cause/runs/20261111T213534Z/` |
| Comparison run 2 (5 arms, tracer records all modes) | `root_cause/runs/20261111T215036Z/` |
| Replication on the original 16-core machine | `root_cause/runs/20261112T032306Z/` |
| 16-core user filter (baseline repeated, tracer on) | `root_cause/runs/20261112T035229Z/` |
| 16-core untraced (baseline, context-aware, single-threaded forest) | `root_cause/runs/20261112T040023Z/` |
| Controls | `root_cause/controls/` |
| Worked example trace | `root_cause/example-trace/` |
| Tool descriptions and commands | `root_cause/README.md` |
| Regression checks (includes the deterministic reproducer) | `run_tests.py` |

Each run folder holds `results.json`, `aggregate.json` (comparison runs),
`summary.jsonl`, `environment.json`, and per attempt `manifest.json`,
`analysis.json`, `stdout_stderr.log` and the compressed raw traces of affected
processes. Traces of unaffected processes were deleted after analysis (about
16 MB per attempt); their per-process summaries are in `analysis.json`.

The workload script changed between the two comparison runs (the
`--user-filter` option was added; the default path is unchanged). Each run's
`environment.json` records the script's hash.
