"""Test whether the capture hook records warnings from joblib workers."""

import json
import os
import warnings
from pathlib import Path

from joblib import Parallel, delayed

TARGET_WARNING = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel`"
)

capture_dir = Path(os.environ["SKLEARN_WARNING_CAPTURE_DIR"])


def emit_target_warning(number):
    warnings.warn(f"{TARGET_WARNING} [probe {number}]", UserWarning)
    return number


if __name__ == "__main__":
    results = Parallel(n_jobs=2)(
        delayed(emit_target_warning)(number) for number in range(4)
    )

    records = []
    for path in capture_dir.glob("capture-*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))

    starts = [r for r in records if r.get("event") == "capture_started"]
    matches = [r for r in records if r.get("event") == "target_warning"]

    print(f"Joblib tasks completed: {len(results)}")
    print(f"Processes with capture-start records: {len(starts)}")
    print(f"Captured target warnings: {len(matches)}")
    print(f"Capture files: {len(list(capture_dir.glob('capture-*.jsonl')))}")

    for record in matches:
        print(
            f"  PID={record.get('pid')}, "
            f"stack_frames={len(record.get('stack', []))}, "
            f"filters_length={record.get('current_warning_filters_length')}"
        )

    if not starts:
        raise SystemExit("FAIL: capture startup was not recorded.")
    if not matches:
        raise SystemExit("FAIL: target warnings were not captured.")
    if any(not record.get("stack") for record in matches):
        raise SystemExit("FAIL: at least one warning has no stack.")

    print("PASS: warning capture recorded matching warnings and stacks.")
