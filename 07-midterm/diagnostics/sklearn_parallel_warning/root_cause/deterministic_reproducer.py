"""Deterministic reproducer for the false-positive scikit-learn warning.

    `sklearn.utils.parallel.delayed` should be used with
    `sklearn.utils.parallel.Parallel` ...

No dataset, no estimator, no randomness, no shortened switch interval. It runs
scikit-learn's own ``Parallel`` and ``delayed`` (correctly paired) on the
threading backend with two threads and four trivial tasks, and fixes the order
in which the two threads enter and leave ``warnings.catch_warnings``. Nothing
about *what* the threads do is changed, only *when* they do it.

Root cause being demonstrated (scikit-learn 1.9.1, ``sklearn/utils/parallel.py``)
---------------------------------------------------------------------------------
1. ``Parallel.__call__`` stores ``warnings._get_filters()`` for the tasks.
   Without context-aware warnings this is the live, process-wide
   ``warnings.filters`` list itself, not a copy.
2. Every task then runs ``with warnings.catch_warnings(): warnings.resetwarnings()``
   and re-adds the stored filters. ``catch_warnings`` swaps the global list for
   a copy on entry and puts the saved list back on exit. ``resetwarnings()``
   empties whichever list is global at that moment.
3. With two threads the swaps can interleave:

       thread A enters   saved = L1 (the stored list)   global = L2
       thread B enters   saved = L2                     global = L3
       thread A exits                                   global = L1
       thread B calls resetwarnings()  ->  empties L1, the stored list

4. Every task that starts afterwards sees an empty stored list, and the check
   ``if not config or not warning_filters`` emits the warning although
   ``Parallel`` and ``delayed`` were paired and the configuration is present.

Usage
-----
    python deterministic_reproducer.py                # expect: tasks 3 and 4 warn
    PYTHON_CONTEXT_AWARE_WARNINGS=1 python deterministic_reproducer.py
                                                      # expect: no warning
    python deterministic_reproducer.py --snapshot     # expect: no warning
        (test-only patch: tasks get a copy of the filter list instead of the
        live list; this is the candidate upstream change, not user advice)

Exit status: 0 if the outcome matches the expectation for the mode, 1 if not.
"""

import argparse
import itertools
import sys
import threading
import warnings

import joblib
import sklearn
import sklearn.utils.parallel as sk_parallel
from sklearn.utils.parallel import Parallel, delayed

TARGET = "`sklearn.utils.parallel.delayed` should be used with"
WAIT_SECONDS = 10

_order_lock = threading.Lock()
_entry_counter = itertools.count()
_second_entered = threading.Event()
_first_exited = threading.Event()
_timeouts = []


class OrderedCatchWarnings(warnings.catch_warnings):
    """``catch_warnings`` with a fixed entry/exit order for the first two threads.

    The first thread to enter does not exit until the second has entered. The
    second thread to enter does not continue (to ``resetwarnings()``) until the
    first has exited. Later entries are not touched.
    """

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._position = None
            return super().__enter__()
        with _order_lock:
            log = super().__enter__()
            self._position = next(_entry_counter)
        if self._position == 1:
            _second_entered.set()
            if not _first_exited.wait(WAIT_SECONDS):
                _timeouts.append("second thread waited for the first to exit")
        return log

    def __exit__(self, *exc_info):
        if self._position == 0:
            if not _second_entered.wait(WAIT_SECONDS):
                _timeouts.append("first thread waited for the second to enter")
        result = super().__exit__(*exc_info)
        if self._position == 0:
            _first_exited.set()
        return result


def task(number):
    """Stand-in for fitting one tree. Returns its task number."""
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--snapshot", action="store_true",
        help="test-only patch: give tasks a copy of the filter list")
    args = parser.parse_args()

    context_aware = bool(getattr(sys.flags, "context_aware_warnings", 0))
    print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} "
          f"| joblib {joblib.__version__}")
    print(f"context_aware_warnings={int(context_aware)} | snapshot patch="
          f"{'on' if args.snapshot else 'off'}")

    if args.snapshot:
        original = sk_parallel._with_config_and_warning_filters
        copies = {}

        def with_snapshot(delayed_func, config, warning_filters):
            key = id(warning_filters)
            if key not in copies:
                copies[key] = (warning_filters, list(warning_filters))
            return original(delayed_func, config, copies[key][1])

        sk_parallel._with_config_and_warning_filters = with_snapshot

    # What Parallel.__call__ will store for the tasks (same expression as
    # sklearn/utils/parallel.py lines 79-82).
    getter = getattr(warnings, "_get_filters", None)
    stored = getter() if getter is not None else warnings.filters
    print(f"\nBefore: {len(stored)} warning filters; the list handed to the "
          f"tasks is the live warnings.filters object: "
          f"{stored is warnings.filters}")

    emitted = []
    original_warn = warnings.warn

    def recording_warn(message, category=None, stacklevel=1, source=None, **kw):
        if TARGET in str(message):
            frame = sys._getframe(1)
            emitted.append({
                "thread": threading.current_thread().name,
                "config_present": bool(frame.f_locals.get("config")),
                "stored_filters": len(frame.f_locals.get("warning_filters", [])),
                "where": f"{frame.f_code.co_filename.split('site-packages/')[-1]}"
                         f":{frame.f_lineno}",
            })
        return original_warn(message, category, stacklevel + 1, source, **kw)

    warnings.warn = recording_warn
    warnings.catch_warnings = OrderedCatchWarnings
    try:
        # pre_dispatch=2: tasks 3 and 4 are handed out only after the first
        # two have finished, as with the later trees of a random forest.
        results = Parallel(n_jobs=2, backend="threading", pre_dispatch=2)(
            delayed(task)(i) for i in range(1, 5))
    finally:
        warnings.catch_warnings = OrderedCatchWarnings.__mro__[1]
        warnings.warn = original_warn

    print(f"After:  {len(stored)} warning filters in the list handed to the "
          f"tasks; results={results}")
    print(f"Target warnings emitted: {len(emitted)}")
    for item in emitted:
        print(f"  {item}")
    if _timeouts:
        print(f"SCHEDULING TIMEOUTS (run is not valid): {_timeouts}")
        return 1

    expect_warning = not context_aware and not args.snapshot
    ok = bool(emitted) == expect_warning
    print(f"\nExpected a warning in this mode: {expect_warning}. "
          f"Observed: {bool(emitted)}. {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
