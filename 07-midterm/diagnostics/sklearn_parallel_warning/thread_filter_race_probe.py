"""Stdlib-only probe of the suspected cause of the scikit-learn warning.

Hypothesis (see FINDINGS.md)
----------------------------
scikit-learn's ``_FuncWrapper.__call__`` (sklearn/utils/parallel.py, 1.9.1,
lines 140-184) runs, in every task::

    with warnings.catch_warnings():
        warnings.resetwarnings()
        ...re-add the filters captured by Parallel.__call__...

When tasks run in threads (joblib "threading" backend, as used by
RandomForestClassifier(n_jobs=-1)) and Python is NOT using context-aware
warnings, ``warnings.filters`` is one process-wide list and ``catch_warnings``
is not thread-safe. Interleaved enter/exit can leave the global filter list
truncated or empty. The *next* ``Parallel.__call__`` then captures that list,
and line 143 (``if not config or not warning_filters``) emits the warning.

This script copies the wrapper's filter handling (BSD-3, scikit-learn
developers) and exercises it with ``ThreadPoolExecutor``. It needs no
scikit-learn, joblib, numpy or dataset. It therefore tests the mechanism, NOT
the real GridSearchCV workload.

The switch interval is shortened to make a rare interleaving frequent. This is
a deliberate amplification; it shows the interleaving is possible, not how
often it happens at the default interval.

Usage
-----
    python thread_filter_race_probe.py
    PYTHON_CONTEXT_AWARE_WARNINGS=1 python thread_filter_race_probe.py
"""

import argparse
import collections
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

KEYS = ["action", "message", "category", "module", "lineno"]
_lock = threading.Lock()
_condition_hits = 0


def current_filters():
    getter = getattr(warnings, "_get_filters", None)
    return list(getter() if getter is not None else warnings.filters)


def wrapped_call(config, warning_filters):
    """Copy of the filter handling in sklearn's _FuncWrapper.__call__."""
    global _condition_hits
    if not config or not warning_filters:  # sklearn parallel.py line 143
        with _lock:
            _condition_hits += 1

    with warnings.catch_warnings():
        warnings.resetwarnings()
        for filter_args in warning_filters:
            spec = {k: v for k, v in zip(KEYS, filter_args) if v is not None}
            if "message" not in spec and "module" not in spec:
                warnings.simplefilter(**spec, append=True)
            else:
                for key in ("message", "module"):
                    value = spec.get(key)
                    if value is not None and not isinstance(value, str):
                        spec[key] = value.pattern
                warnings.filterwarnings(**spec, append=True)
        sum(range(200))  # stand-in for the task body


def one_trial(threads, tasks, rounds):
    """Run rounds like consecutive Parallel calls; stop once corrupted."""
    for _ in range(rounds):
        captured = current_filters()  # what Parallel.__call__ captures
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda _: wrapped_call({"k": 1}, captured), range(tasks)))
        if len(current_filters()) != len(captured):
            return


def main():
    global _condition_hits
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--switch-interval", type=float, default=1e-6,
                    help="seconds; CPython default is 0.005")
    args = ap.parse_args()

    sys.setswitchinterval(args.switch_interval)
    print(f"Python {sys.version.split()[0]} | "
          f"context_aware_warnings={getattr(sys.flags, 'context_aware_warnings', 'n/a')} | "
          f"switch_interval={args.switch_interval}")

    distribution = collections.Counter()
    next_call_warns = 0
    for _ in range(args.trials):
        baseline = current_filters()
        one_trial(args.threads, args.tasks, args.rounds)
        distribution[len(current_filters())] += 1

        # The "next Parallel call": capture whatever the global list is now.
        _condition_hits = 0
        captured = current_filters()
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: wrapped_call({"k": 1}, captured), range(10)))
        next_call_warns += bool(_condition_hits)

        warnings.resetwarnings()  # restore for the next trial
        for item in baseline:
            warnings.filters.append(item)

    print(f"filters at end (started with {len(baseline)}): "
          f"{dict(sorted(distribution.items()))}")
    print(f"trials where the next Parallel call would emit the warning: "
          f"{next_call_warns}/{args.trials}")


if __name__ == "__main__":
    main()
