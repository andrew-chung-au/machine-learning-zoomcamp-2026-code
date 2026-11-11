"""Record which list object every warning-filter operation acts on.

Purpose
-------
The earlier tracker (``capture.TrackingFilters``) replaced ``warnings.filters``
with a list subclass and was lost the first time ``catch_warnings`` installed a
plain copy. This tracer does not depend on the list object. It wraps the three
operations that scikit-learn's ``_FuncWrapper.__call__`` performs and records
the *identity* (``id()``) and length of the list involved each time:

    E  catch_warnings.__enter__   list saved, list installed, and (when the
                                  caller is _FuncWrapper.__call__) the list the
                                  task was handed by Parallel.__call__
    R  warnings.resetwarnings()   list emptied, and its length just before
    X  catch_warnings.__exit__    list put back as the global list
    W  the target warning         list the task was handed, and the global list

The decisive record is an ``R`` whose list is one that tasks were handed and
whose length before was non-zero: a thread emptying the stored filter list.

"The list" is always the one in force for the calling thread: the process-wide
``warnings.filters`` without context-aware warnings, the current context's list
with them. The same records are therefore written in both modes, and the
length of the list each task was handed is measured in both.

Design points
-------------
- Each operation and its record are made under ``warnings._lock`` (a re-entrant
  lock that the wrapped operations take anyway), so a record cannot describe a
  different list from the one the operation used.
- Records are appended to an in-memory list (no I/O, so no extra points where
  another thread can take over). They are written to
  ``$FILTER_TRACE_DIR/trace-<pid>.jsonl`` when the main thread leaves a
  ``catch_warnings`` block (no worker threads are running then) and at exit.
- Installed scikit-learn is not modified. Tracing is inactive unless
  ``FILTER_TRACE_DIR`` is set.
- Optional ``FILTER_TRACE_SNAPSHOT=1`` applies the candidate upstream change
  (tasks get a copy of the filter list, not the live list) so its effect can be
  measured. This is a test arm, not advice for users.

The tracer adds a little work to every task, so it can change how often the
race happens. It cannot make an operation act on a different list.
"""

import atexit
import json
import os
import sys
import threading
import time
import warnings

TARGET_WARNING = "`sklearn.utils.parallel.delayed` should be used with"
WRAPPER_FILE_SUFFIX = os.path.join("sklearn", "utils", "parallel.py")

_events = []
_flush_lock = threading.Lock()
_installed = False
_main_ident = threading.main_thread().ident
_original_catch = warnings.catch_warnings
_original_reset = warnings.resetwarnings
_original_warn = warnings.warn


def _current():
    """The filter list in force for this thread.

    Without context-aware warnings this is the process-wide
    ``warnings.filters``. With them it is the list of the current context
    (``warnings.filters`` is then not what ``catch_warnings`` swaps).
    """
    getter = getattr(warnings, "_get_filters", None)
    return getter() if getter is not None else warnings.filters


def _trace_dir():
    return os.environ.get("FILTER_TRACE_DIR")


def _flush():
    directory = _trace_dir()
    if not directory or not _events:
        return
    with _flush_lock:
        pending = _events[:]
        del _events[:len(pending)]
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"trace-{os.getpid()}.jsonl")
            with open(path, "a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(e) + "\n" for e in pending))
        except Exception as exc:
            print(f"FILTER_TRACE_WRITE_ERROR: {exc!r}", file=sys.stderr, flush=True)


class TracedCatchWarnings(_original_catch):
    def __enter__(self):
        handed = None
        frame = sys._getframe(1)
        if (frame.f_code.co_name == "__call__"
                and frame.f_code.co_filename.endswith(WRAPPER_FILE_SUFFIX)):
            handed = frame.f_locals.get("warning_filters")
        with warnings._lock:
            saved = _current()
            saved_len = len(saved)
            log = super().__enter__()
            _events.append({
                "k": "E", "t": time.monotonic_ns(), "tid": threading.get_ident(),
                "saved": id(saved), "saved_len": saved_len,
                "installed": id(_current()),
                "handed": id(handed) if handed is not None else None,
                "handed_len": len(handed) if handed is not None else None,
                "handed_is_saved": handed is saved if handed is not None else None,
            })
        return log

    def __exit__(self, *exc_info):
        with warnings._lock:
            leaving = _current()
            result = super().__exit__(*exc_info)
            _events.append({
                "k": "X", "t": time.monotonic_ns(), "tid": threading.get_ident(),
                "left": id(leaving), "left_len": len(leaving),
                "restored": id(_current()),
                "restored_len": len(_current()),
            })
        if threading.get_ident() == _main_ident:
            _flush()
        return result


def traced_reset():
    with warnings._lock:
        current = _current()
        before = len(current)
        _original_reset()
        _events.append({
            "k": "R", "t": time.monotonic_ns(), "tid": threading.get_ident(),
            "list": id(current), "len_before": before,
        })


def traced_warn(message, category=None, stacklevel=1, source=None, **kwargs):
    if TARGET_WARNING in str(message):
        frame = sys._getframe(1)
        handed = frame.f_locals.get("warning_filters")
        current = _current()
        _events.append({
            "k": "W", "t": time.monotonic_ns(), "tid": threading.get_ident(),
            "handed": id(handed) if handed is not None else None,
            "handed_len": len(handed) if handed is not None else None,
            "config_present": bool(frame.f_locals.get("config")),
            "current": id(current), "current_len": len(current),
            "emitter": f"{frame.f_code.co_filename}:{frame.f_lineno}",
            "thread_is_main": threading.get_ident() == _main_ident,
        })
    return _original_warn(message, category, stacklevel + 1, source, **kwargs)


def _install_snapshot_patch():
    """Candidate upstream change: hand tasks a copy of the filter list.

    Re-implements ``sklearn.utils.parallel.Parallel.__call__`` (1.9.1) with one
    difference: ``list(...)`` around the captured filters.
    """
    import joblib
    import sklearn.utils.parallel as sk_parallel
    from sklearn._config import get_config

    def call_with_snapshot(self, iterable):
        config = get_config()
        filters_func = getattr(warnings, "_get_filters", None)
        warning_filters = list(
            filters_func() if filters_func is not None else warnings.filters
        )
        iterable_with_config_and_warning_filters = (
            (
                sk_parallel._with_config_and_warning_filters(
                    delayed_func, config, warning_filters),
                args,
                kwargs,
            )
            for delayed_func, args, kwargs in iterable
        )
        return joblib.Parallel.__call__(
            self, iterable_with_config_and_warning_filters)

    sk_parallel.Parallel.__call__ = call_with_snapshot


def install():
    global _installed
    if _installed or not _trace_dir():
        return
    _installed = True
    snapshot = os.environ.get("FILTER_TRACE_SNAPSHOT") == "1"
    _events.append({
        "k": "START", "t": time.monotonic_ns(), "pid": os.getpid(),
        "python": sys.version.split()[0],
        "context_aware_warnings": getattr(sys.flags, "context_aware_warnings", None),
        "initial_filters": len(_current()),
        "trace_version": 2,
        "snapshot_patch": snapshot,
        "argv0": sys.argv[0] if sys.argv else None,
    })
    warnings.catch_warnings = TracedCatchWarnings
    warnings.resetwarnings = traced_reset
    warnings.warn = traced_warn
    if snapshot:
        _install_snapshot_patch()
    atexit.register(_flush)
    _flush()
