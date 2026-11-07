import inspect
import os
import sys
import traceback
import warnings
from pathlib import Path

TARGET_WARNING = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel`"
)

_original_warn = warnings.warn
_stack_saved = False


def traced_warn(message, category=None, stacklevel=1, source=None, **kwargs):
    global _stack_saved

    if TARGET_WARNING in str(message) and not _stack_saved:
        _stack_saved = True

        log_path = Path(
            os.environ.get(
                "SKLEARN_WARNING_STACK_LOG",
                "sklearn_warning_stack.log",
            )
        )

        # State in the frame that emitted the warning.
        frame = inspect.currentframe().f_back
        config = frame.f_locals.get("config")
        warning_filters = frame.f_locals.get("warning_filters")

        # Current warning-filter state in the worker.
        get_filters = getattr(warnings, "_get_filters", None)
        current_filters = (
            get_filters() if get_filters is not None else warnings.filters
        )

        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nWARNING PID={os.getpid()}\n")
            log.write(
                f"context_aware_warnings="
                f"{getattr(sys.flags, 'context_aware_warnings', 'unavailable')}\n"
            )
            log.write(f"config={config!r}\n")
            log.write(
                f"passed_warning_filters_length="
                f"{len(warning_filters) if warning_filters is not None else 'None'}\n"
            )
            log.write(
                f"worker_warnings_filters_length={len(warnings.filters)}\n"
            )
            log.write(
                f"worker_get_filters_length={len(current_filters)}\n"
            )
            log.write(f"worker_get_filters={current_filters!r}\n")
            traceback.print_stack(file=log)
            log.write("END WARNING STACK\n")

    return _original_warn(
        message,
        category=category,
        stacklevel=stacklevel,
        source=source,
        **kwargs,
    )


warnings.warn = traced_warn
