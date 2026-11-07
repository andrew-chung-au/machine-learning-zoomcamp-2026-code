"""Validate warning-capture evidence and compare state per process."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_records(directory):
    records = []
    problems = []

    for path in sorted(directory.glob("capture-*.jsonl")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                record = json.loads(line)
                record["_capture_file"] = path.name
                records.append(record)
            except json.JSONDecodeError:
                problems.append(f"{path.name}:{number}: invalid JSON")

    return records, problems


def summarise(directory):
    records, problems = load_records(directory)
    starts = defaultdict(list)
    warnings = defaultdict(list)

    for record in records:
        pid = record.get("pid")
        if pid is None:
            problems.append("Record missing PID")
            continue

        if record.get("event") == "capture_started":
            starts[pid].append(record)
        elif record.get("event") == "target_warning":
            warnings[pid].append(record)

    print(f"\nEvidence: {directory}")
    print(f"Capture files: {len(list(directory.glob('capture-*.jsonl')))}")
    print(f"Startup records: {sum(map(len, starts.values()))}")
    print(f"Target warnings: {sum(map(len, warnings.values()))}")
    print(f"Warning-emitting PIDs: {len(warnings)}")

    for pid, events in sorted(warnings.items()):
        startup = starts.get(pid, [])
        first = events[0]

        print(f"\nPID {pid}: {len(events)} warning(s)")
        if not startup:
            print("  Startup state: MISSING — cannot establish a transition")
        else:
            initial = startup[0].get("initial_warning_filters")
            initial_length = (
                len(initial) if isinstance(initial, (list, tuple)) else None
            )
            print(f"  Startup filter count: {initial_length}")
            print(f"  Startup context-aware warnings: "
                  f"{startup[0].get('context_aware_warnings')}")

        passed_counts = Counter(
            event.get("passed_warning_filters_length") for event in events
        )
        current_counts = Counter(
            event.get("current_warning_filters_length") for event in events
        )
        config_empty = Counter(
            event.get("captured_config_empty") for event in events
        )

        print(f"  Passed-filter lengths: {dict(passed_counts)}")
        print(f"  Current-filter lengths: {dict(current_counts)}")
        print(f"  Config-empty values: {dict(config_empty)}")
        print(f"  Warning emitter: {first.get('emitter_file')}:"
              f"{first.get('emitter_line')}")

    return records, starts, warnings, problems


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python validate_evidence.py EVIDENCE_DIRECTORY")

    root = Path(sys.argv[1]).resolve()
    problems = []

    if not root.is_dir():
        raise SystemExit(f"Evidence directory does not exist: {root}")

    # Accept a single probe directory or a matrix-run directory.
    if list(root.glob("capture-*.jsonl")):
        directories = [root]
    else:
        directories = sorted(root.glob("grid-*_forest-*__repeat-*"))

    if not directories:
        raise SystemExit(f"No capture files or matrix attempts found under {root}")

    total_warnings = 0
    total_starts = 0

    for directory in directories:
        capture_dir = directory / "capture" if (directory / "capture").is_dir() else directory
        records, starts, warnings, issues = summarise(capture_dir)
        problems.extend(f"{directory.name}: {issue}" for issue in issues)
        total_warnings += sum(map(len, warnings.values()))
        total_starts += sum(map(len, starts.values()))

        for pid in warnings:
            if pid not in starts:
                problems.append(
                    f"{directory.name}: PID {pid} emitted warnings "
                    "but has no startup record"
                )

        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("target_warning_records") != sum(map(len, warnings.values())):
                    problems.append(
                        f"{directory.name}: manifest warning count mismatch"
                    )
                if manifest.get("capture_processes_started") != sum(map(len, starts.values())):
                    problems.append(
                        f"{directory.name}: manifest startup count mismatch"
                    )
            except Exception as exc:
                problems.append(f"{directory.name}: unreadable manifest: {exc}")

    print("\n===== VALIDATION SUMMARY =====")
    print(f"Directories checked: {len(directories)}")
    print(f"Startup records: {total_starts}")
    print(f"Target warnings: {total_warnings}")

    if problems:
        print("\nVALIDATION ISSUES:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\nCapture records are structurally consistent.")
    print("This does not, by itself, establish the cause of the warning.")


if __name__ == "__main__":
    main()
