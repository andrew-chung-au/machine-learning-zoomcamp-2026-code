"""Capture target scikit-learn warnings in each Python process as JSONL."""

import json
import os
import sys
import threading
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

TARGET_WARNING = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel`"
)

_lock = threading.Lock()
_original_warn = warnings.warn
_installed = False


def _filters():
    getter = getattr(warnings, "_get_filters", None)
    try:
        return getter() if getter is not None else warnings.filters
    except Exception as exc:
        return {"capture_error": repr(exc)}


def _write(event):
    """Append one complete JSON record to this process's own file."""
    directory = os.environ.get("SKLEARN_WARNING_CAPTURE_DIR")
    if not directory:
        return

    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        event.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        event.setdefault("pid", os.getpid())
        event.setdefault("python", sys.version)
        event.setdefault(
            "context_aware_warnings",
            getattr(sys.flags, "context_aware_warnings", None),
        )

        target = path / f"capture-{os.getpid()}.jsonl"
        line = json.dumps(event, default=str, sort_keys=True)

        # Each process has its own file; the lock protects threads in that process.
        with _lock:
            with target.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
    except Exception as exc:
        # Instrumentation must not silently break the model being tested.
        try:
            print(
                f"WARNING_CAPTURE_ERROR: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass


def traced_warn(message, category=None, stacklevel=1, source=None, **kwargs):
    """Record every call carrying the target message, before normal filtering."""
    text = str(message)

    if TARGET_WARNING in text:
        frame = sys._getframe(1)
        config = frame.f_locals.get("config")
        passed_filters = frame.f_locals.get("warning_filters")
        current_filters = _filters()
        actual_filters = warnings.filters

        global _filter_replacement_recorded
        if (
            os.environ.get("SKLEARN_WARNING_TRACE_FILTERS") == "1"
            and not isinstance(actual_filters, TrackingFilters)
            and not _filter_replacement_recorded
        ):
            _filter_replacement_recorded = True
            _write({
                "event": "warning_filters_replaced_or_untracked",
                "actual_filters_type": type(actual_filters).__name__,
                "actual_filters_length": (
                    len(actual_filters)
                    if isinstance(actual_filters, (list, tuple))
                    else None
                ),
                "getter_filters_length": (
                    len(current_filters)
                    if isinstance(current_filters, (list, tuple))
                    else None
                ),
                "stack": traceback.format_stack(frame, limit=20),
            })

        _write({
            "event": "target_warning",
            "message": text,
            "category": getattr(category, "__name__", str(category)),
            "emitter_file": frame.f_code.co_filename,
            "emitter_line": frame.f_lineno,
            "emitter_function": frame.f_code.co_name,
            "captured_config": config,
            "captured_config_empty": not bool(config),
            "passed_warning_filters": passed_filters,
            "passed_warning_filters_length": (
                len(passed_filters) if passed_filters is not None else None
            ),
            "passed_warning_filters_empty": (
                not bool(passed_filters) if passed_filters is not None else None
            ),
            "current_warning_filters": current_filters,
            "current_warning_filters_length": (
                len(current_filters)
                if isinstance(current_filters, (list, tuple))
                else None
            ),
            "stack": traceback.format_stack(frame, limit=40),
        })

    return _original_warn(
        message,
        category=category,
        stacklevel=stacklevel,
        source=source,
        **kwargs,
    )




class TrackingFilters(list):
    """Record the first transition from nonempty to empty, when enabled."""

    def _check_transition(self, before):
        global _filter_transition_recorded

        if (
            os.environ.get("SKLEARN_WARNING_TRACE_FILTERS") == "1"
            and before > 0
            and len(self) == 0
            and not _filter_transition_recorded
        ):
            _filter_transition_recorded = True
            _write({
                "event": "warning_filters_became_empty",
                "previous_length": before,
                "new_length": len(self),
                "context_aware_warnings": getattr(
                    sys.flags, "context_aware_warnings", None
                ),
                "stack": traceback.format_stack(limit=20),
            })

    def _mutate(self, operation, *args):
        before = len(self)
        result = operation(*args)
        self._check_transition(before)
        return result

    def append(self, item):
        return self._mutate(super().append, item)

    def clear(self):
        return self._mutate(super().clear)

    def extend(self, items):
        return self._mutate(super().extend, items)

    def insert(self, index, item):
        return self._mutate(super().insert, index, item)

    def pop(self, *args):
        return self._mutate(super().pop, *args)

    def remove(self, item):
        return self._mutate(super().remove, item)

    def reverse(self):
        return self._mutate(super().reverse)

    def sort(self, *args, **kwargs):
        before = len(self)
        result = super().sort(*args, **kwargs)
        self._check_transition(before)
        return result

    def __setitem__(self, key, value):
        return self._mutate(super().__setitem__, key, value)

    def __delitem__(self, key):
        return self._mutate(super().__delitem__, key)

    def __iadd__(self, items):
        before = len(self)
        super().__iadd__(items)
        self._check_transition(before)
        return self

    def __imul__(self, count):
        before = len(self)
        super().__imul__(count)
        self._check_transition(before)
        return self


_filter_transition_recorded = False
_filter_replacement_recorded = False


def install():
    global _installed
    if _installed:
        return
    _installed = True
    warnings.warn = traced_warn

    if os.environ.get("SKLEARN_WARNING_TRACE_FILTERS") == "1":
        # Preserve the existing filters while tracking list mutations.
        warnings.filters = TrackingFilters(warnings.filters)

    _write({
        "event": "capture_started",
        "capture_version": 1,
        "target_warning": TARGET_WARNING,
        "python_executable": sys.executable,
        "python_path": sys.path,
        "pythonwarnings_env": os.environ.get("PYTHONWARNINGS"),
        "capture_directory": os.environ.get("SKLEARN_WARNING_CAPTURE_DIR"),
        "initial_warning_filters": _filters(),
    })


install()
