**Question**

How do I use `GridSearchCV` with parallel processing, and how do I fix the scikit-learn warning "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"?

**Answer**

**Put `n_jobs=-1` on `GridSearchCV` and leave the model itself single-threaded: do not also pass `n_jobs=-1` to `RandomForestClassifier`. The warning appears when both are parallel. It is then a false alarm caused by a thread-safety bug in scikit-learn, not by your code, and it does not mean your results are wrong. A `warnings.filterwarnings` filter does not reliably hide it.**

`GridSearchCV` tests different combinations of model parameters using cross-validation, then identifies the combination with the best score. Parallel processing runs several model fits at once, which can make the search much faster.

A self-contained example (every name is defined):

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X_train, y_train = make_classification(
    n_samples=2000, n_features=30, random_state=1
)

# The forest is left single-threaded (n_jobs is not set).
est = RandomForestClassifier(random_state=1)
grid = {
    "n_estimators": [200],
    "max_depth": [10, 20, None],
    "max_features": ["sqrt", 0.5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# The grid search runs the 30 fits in parallel, one per CPU core.
gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
```

**The warning**

```text
UserWarning: `sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel` to make it possible to propagate
the scikit-learn configuration of the current thread to the joblib workers.
```

**Why it appears**

There are two different situations.

- **Your own code calls `joblib.Parallel` or `joblib.delayed` together with scikit-learn's versions.** Then the message is accurate. Import both `Parallel` and `delayed` from `sklearn.utils.parallel`, as the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel.delayed.html) says.
- **Your code only uses `GridSearchCV` and a model with `n_jobs=-1`, such as `RandomForestClassifier(n_jobs=-1)`.** Then the message is wrong. `GridSearchCV(n_jobs=-1)` starts several worker processes, and inside each one the forest runs its trees in several threads. In every task scikit-learn empties Python's list of warning filters and fills it again. Without Python's context-aware warnings that list is shared by all threads in a process, so the threads overwrite each other's changes and the list can end up empty. scikit-learn then mistakes "the filter list is empty" for "`delayed` was used without `Parallel`" and warns.

In the second situation the warning is intermittent (a clean run does not mean the problem is absent), scikit-learn's configuration still reaches the workers, and the best parameters were the same in every test run. The bug is tracked as [scikit-learn issue #35089](https://github.com/scikit-learn/scikit-learn/issues/35089).

**How to fix it**

**Option 1 (recommended, any Python version): parallelise at one level only.**

Keep `n_jobs=-1` on `GridSearchCV` and remove `n_jobs` from the model, as in the example above. Without threads inside the workers there is nothing to overwrite. The grid search still uses every core. In the tests this was not slower than making both parallel (10.5 s against 11.1 s on a 16-core machine, 12.7 s against 13.1 s on a 4-core machine).

If you fit a single model without a grid search, `n_jobs=-1` on the model alone is fine. The warning was not seen in that layout in 8 test runs.

**Option 2 (Python 3.14 or later): enable context-aware warnings when Python starts.**

Use this if you need both levels of parallelism. It gives each thread its own filter list.

For JupyterLab, run this from your project directory:

```bash
PYTHON_CONTEXT_AWARE_WARNINGS=1 uv run jupyter lab
```

For Jupyter Notebook:

```bash
PYTHON_CONTEXT_AWARE_WARNINGS=1 uv run jupyter notebook
```

Stop the existing Jupyter server first, then restart it with the command above. The setting must be present when Python starts; setting it inside a running notebook is not equivalent. To check it:

```python
import sys

print(sys.flags.context_aware_warnings)
```

`1` means it is enabled. If the attribute does not exist, your Python is older than 3.14 and this option does not apply.

**What does not work: a warning filter.**

```python
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
```

This is the usual first attempt, and it fails here for the same reason the warning appears: the filter is one of the entries that gets lost. In the tests 12 of 15 runs still printed the warning with this filter in place.

**Evidence**

`GridSearchCV` over `RandomForestClassifier` on the [UCI Phishing Websites dataset](https://archive.ics.uci.edu/dataset/327/phishing), a fresh Python process for every run, 16 worker processes with 16 forest threads each. Machine A has 16 cores and used plain `n_jobs=-1`; machine B has 4 cores and was given the same layout explicitly.

| Setup | Runs with the warning, machine A | Machine B |
| --- | --- | --- |
| Grid search and forest both parallel | 13 of 15 | 21 of 30 |
| Option 1: forest single-threaded | 0 of 15 | 0 of 30 |
| Option 2: context-aware warnings | 0 of 15 | 0 of 30 |
| Both parallel, with the warning filter above | 11 of 15 | 12 of 15 |

Every one of the 12,690 warnings in the first row was traced to a recorded event in which a thread emptied a filter list that other tasks were using. A [one-second reproducer](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/root_cause/deterministic_reproducer.py) triggers the same event on every run using only scikit-learn's own `Parallel` and `delayed`, with no dataset. The method, data and limits are in the [root-cause report](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/ROOT_CAUSE.md).

**Environment and limits**

- Python 3.14.5 (machine A) and 3.14.6 (machine B), standard GIL-enabled builds; scikit-learn 1.9.1 (the latest release when tested), joblib 1.6.0, NumPy 2.5.4, SciPy 1.17.1.
- Two machines, one dataset and one parameter grid.
- How often the warning appears depends on the number of processes and threads. With 4 processes of 4 threads each it did not appear in 10 runs.
- Options 1 and 2 avoid the conditions the bug needs. They do not change scikit-learn, and a later scikit-learn release may make them unnecessary.

**If the warning persists**

Check that no model inside the grid search has `n_jobs` set, then check your package versions. If you report it, include the estimator and parameter grid, the exact command, the full warning text with traceback, your Python and scikit-learn versions, and whether your own code uses `joblib` or scikit-learn's `Parallel` / `delayed`.

**References**

- [scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn: `sklearn.utils.parallel.delayed`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel.delayed.html)
- [scikit-learn: Parallelism and resource management](https://scikit-learn.org/stable/computing/parallelism.html)
- [scikit-learn issue #35089: thread-based `Parallel` can corrupt the process-wide warning filters](https://github.com/scikit-learn/scikit-learn/issues/35089)
- [Python: `warnings` module](https://docs.python.org/3/library/warnings.html)
- [Python: Command-line and environment settings](https://docs.python.org/3/using/cmdline.html)
