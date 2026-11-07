**Question**

How do I use `GridSearchCV` with parallel processing, and what should I do about the scikit-learn warning "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"?

**Answer**

**Use `n_jobs=-1` to run grid-search jobs in parallel. If the warning appears, it does not mean your results are wrong. In the setup tested here it came from scikit-learn's own thread handling, and starting Python with `PYTHON_CONTEXT_AWARE_WARNINGS=1` removed it. Treat that as a workaround for a tested setup, not a general fix.**

`GridSearchCV` tests different combinations of model parameters using cross-validation, then identifies the combination with the best score. Parallel processing runs several model fits at once, which can make the search much faster.

A self-contained example (every name is defined):

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X_train, y_train = make_classification(
    n_samples=2000, n_features=30, random_state=1
)

est = RandomForestClassifier(random_state=1, n_jobs=-1)
grid = {
    "n_estimators": [200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1],
    "max_features": ["sqrt", 0.5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
```

The warning was reproduced with the [UCI Phishing Websites dataset](https://archive.ics.uci.edu/dataset/327/phishing) (see "Evidence" below), not with the synthetic data above, and it is **intermittent**: a clean run does not mean the problem is absent. If you use the example above and see no warning, that is expected in some runs.

**What the warning says, and what it can mean**

```text
UserWarning: `sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel` to make it possible to propagate
the scikit-learn configuration of the current thread to the joblib workers.
```

- **If your own code calls `joblib.Parallel` or `joblib.delayed` alongside scikit-learn's `delayed`**, the message is probably accurate. The scikit-learn documentation says `sklearn.utils.parallel.delayed` should be paired with `sklearn.utils.parallel.Parallel` when you need scikit-learn configuration passed to the workers. Use both from `sklearn.utils.parallel`.
- **If your code only uses `GridSearchCV` and `RandomForestClassifier`**, as in the example above, the warning came from inside scikit-learn. In the tested setup, scikit-learn's check fired because the list of Python warning filters it had captured was empty, even though its configuration was present. The message wording does not describe that situation.

Either way, the warning concerns how scikit-learn's configuration reaches the workers. It does not by itself indicate a problem with your data, model, or ROC AUC.

**Likely cause (not directly observed)**

Without Python's context-aware warnings, the warning filters are shared by all threads in a process, and `warnings.catch_warnings` is not thread-safe. A random forest with `n_jobs=-1` runs tasks in threads inside the grid-search workers, and scikit-learn resets and restores the filters in every task. When threads overlap, the shared filter list can end up empty, and the next parallel call reads the empty list and warns. The same mechanism is described in [scikit-learn issue #35089](https://github.com/scikit-learn/scikit-learn/issues/35089). This explanation fits the evidence below, but the moment the list is emptied has not been recorded, so it remains a hypothesis.

**What to try**

**Step 1: A targeted warning filter (the usual first attempt)**

```python
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
```

Run the grid search again. **This did not work in the tested setup**, and the likely reason is the mechanism above: when the captured filter list is empty, your filter is not re-applied in the worker. It may work in other setups; that was not tested. If the warning still appears, continue to Step 2.

**Step 2: Enable context-aware warnings when Python starts (Python 3.14 and later)**

For JupyterLab, run this from your project directory:

```bash
PYTHON_CONTEXT_AWARE_WARNINGS=1 uv run jupyter lab
```

For Jupyter Notebook:

```bash
PYTHON_CONTEXT_AWARE_WARNINGS=1 uv run jupyter notebook
```

Stop the existing Jupyter server first, then restart it with the command above, using the same project environment as before.

**Important:** the setting must be present when Python starts. Setting it inside an already-running notebook is not equivalent.

To check that it is enabled:

```python
import sys

print(sys.flags.context_aware_warnings)
```

`1` means it is enabled. If this attribute does not exist, your Python is older than 3.14 and this step does not apply.

**Step 3 (observation, not a recommendation): single-threaded forest**

In the logged runs, using `n_jobs=1` on the `RandomForestClassifier` while keeping `n_jobs=-1` on `GridSearchCV` produced no warning in 6 of 6 attempts with context-aware warnings off. This also changes where the parallelism happens, so speed may differ, and it was not tested as a fix.

**Evidence**

Controlled comparison on the [UCI Phishing Websites dataset](https://archive.ics.uci.edu/dataset/327/phishing) (`GridSearchCV(n_jobs=-1)` over `RandomForestClassifier(n_jobs=-1)`, 15 alternating attempts per setting, a fresh Python process each time):

| `PYTHON_CONTEXT_AWARE_WARNINGS` | attempts with the warning |
| --- | --- |
| 0 (off) | 8 of 15 |
| 1 (on) | 0 of 15 |

Fisher exact test, two-sided: p = 0.0022. In every recorded warning, scikit-learn's configuration was present and the captured filter list was empty. The full data, scripts and limitations are in the [investigation report](https://github.com/andrew-chung-au/machine-learning-zoomcamp-2026-code/blob/main/07-midterm/diagnostics/sklearn_parallel_warning/FINDINGS.md).

**Environment used**

- Python: 3.14.5 (standard GIL-enabled build, `Py_GIL_DISABLED = 0`)
- scikit-learn: 1.9.1
- joblib: 1.6.0
- NumPy: 2.5.4, SciPy: 1.17.1

**Limits of this advice**

- Tested on one machine and one software stack, with 30 attempts in the controlled comparison. Results may differ on other systems, Python versions or scikit-learn versions.
- The cause is a probable explanation, not a confirmed one.
- The workaround removed the warning in the tested setup. It is not guaranteed elsewhere, and it does not change your model results.

**If the warning persists**

Check your package versions, and report the following (a minimal report lets others identify the cause): the estimator and parameter grid, the exact command, the full warning text with traceback, your Python and scikit-learn versions, and whether your own code uses `joblib` or scikit-learn's `Parallel` / `delayed`.

**References**

- [scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn: `sklearn.utils.parallel.delayed`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel.delayed.html)
- [scikit-learn: Parallelism and resource management](https://scikit-learn.org/stable/computing/parallelism.html)
- [scikit-learn issue #35089: thread-based `Parallel` can corrupt the process-wide warning filters](https://github.com/scikit-learn/scikit-learn/issues/35089)
- [Python: `warnings` module](https://docs.python.org/3/library/warnings.html)
- [Python: Command-line and environment settings](https://docs.python.org/3/using/cmdline.html)
