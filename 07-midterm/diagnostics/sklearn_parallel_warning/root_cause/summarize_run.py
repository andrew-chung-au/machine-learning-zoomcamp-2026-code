"""Aggregate the per-attempt analyses of one ``compare_fixes.py`` run, per arm.

Reads ``attempt-*/manifest.json`` and ``attempt-*/analysis.json`` in a run
folder, prints one block per arm and writes ``aggregate.json`` next to them.

    uv run python summarize_run.py runs/<run id>
"""

import collections
import json
import sys
from pathlib import Path


def summarize(run_root):
    arms = collections.OrderedDict()
    for manifest_path in sorted(run_root.glob("attempt-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis_path = manifest_path.with_name("analysis.json")
        analysis = (json.loads(analysis_path.read_text(encoding="utf-8"))
                    if analysis_path.exists() else {})
        arm = arms.setdefault(manifest["arm"], {
            "attempts": 0, "attempts_with_warning": 0,
            "attempts_with_warning_shown": 0,
            "attempts_with_filter_corruption": 0,
            "attempts_with_decisive_reset": 0,
            "processes_traced": 0, "processes_with_warning": 0,
            "wrapper_tasks_in_threads": 0, "inner_calls": 0,
            "target_warnings": 0, "target_warning_lines_shown": 0,
            "target_warnings_config_present": 0,
            "warnings_accounted_for": 0, "outer_task_warnings": 0,
            "warnings_where_handed_list_was_global": 0,
            "calls_by_cause": collections.Counter(),
            "warnings_by_cause": collections.Counter(),
            "warning_handed_len": collections.Counter(),
            "warning_global_len": collections.Counter(),
            "warnings_per_process": collections.Counter(),
            "tasks_with_shortened_list_no_warning": 0,
            "decisive_resets": 0,
            "decisive_resets_in_expected_order": 0,
            "decisive_resets_list_made_global_by_other_thread": 0,
            "decisive_reset_enter_to_reset_us": [],
            "captures_of_empty_or_shortened_list": 0,
            "captures_left_global_by_worker_thread_exit": 0,
            "captures_list_created_by_worker_thread": 0,
            "usual_filter_list_length": collections.Counter(),
            "wall_seconds": [],
        })
        arm["attempts"] += 1
        arm["attempts_with_warning"] += manifest["status"] == "warning_observed"
        arm["attempts_with_warning_shown"] += bool(
            manifest["target_warning_lines_in_log"])
        arm["attempts_with_filter_corruption"] += bool(
            manifest["filter_corruption_observed"])
        arm["attempts_with_decisive_reset"] += bool(manifest["decisive_resets"])
        arm["target_warning_lines_shown"] += manifest["target_warning_lines_in_log"]
        arm["wall_seconds"].append(manifest["wall_seconds"])
        if not analysis:
            continue
        for key in ("processes_with_warning", "wrapper_tasks_in_threads",
                    "inner_calls", "target_warnings",
                    "target_warnings_config_present", "warnings_accounted_for",
                    "outer_task_warnings", "warnings_where_handed_list_was_global",
                    "tasks_with_shortened_list_no_warning"):
            arm[key] += analysis[key]
        arm["processes_traced"] += analysis["trace_files"]
        arm["usual_filter_list_length"][str(analysis["usual_filter_list_length"])] += 1
        for key in ("calls_by_cause", "warnings_by_cause", "warning_handed_len",
                    "warning_global_len"):
            arm[key].update(analysis[key])
        for process in analysis["processes"]:
            if process["warnings"]:
                arm["warnings_per_process"][str(process["warnings"])] += 1
        for chain in analysis["decisive_resets"]:
            arm["decisive_resets"] += 1
            arm["decisive_resets_in_expected_order"] += bool(
                chain["order_enter_then_exit_then_reset"])
            arm["decisive_resets_list_made_global_by_other_thread"] += bool(
                chain["made_global_by_other_thread"])
            if chain["microseconds_enter_to_reset"] is not None:
                arm["decisive_reset_enter_to_reset_us"].append(
                    round(chain["microseconds_enter_to_reset"]))
        for capture in analysis["captures"]:
            arm["captures_of_empty_or_shortened_list"] += 1
            arm["captures_left_global_by_worker_thread_exit"] += bool(
                capture["left_global_by_worker_thread_exit"])
            arm["captures_list_created_by_worker_thread"] += bool(
                capture["created_by_worker_thread"])

    for arm in arms.values():
        for key, value in list(arm.items()):
            if isinstance(value, collections.Counter):
                arm[key] = dict(sorted(value.items(), key=lambda kv: int(kv[0])
                                       if kv[0].lstrip("-").isdigit() else 0))
        arm["decisive_reset_enter_to_reset_us"].sort()
        walls = sorted(arm.pop("wall_seconds"))
        arm["median_wall_seconds"] = walls[len(walls) // 2] if walls else None
    return arms


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_root = Path(sys.argv[1]).resolve()
    arms = summarize(run_root)
    if not arms:
        print(f"No attempts found in {run_root}")
        return 1
    for name, arm in arms.items():
        print(f"\n[{name}]")
        for key, value in arm.items():
            print(f"  {key}: {value}")
    (run_root / "aggregate.json").write_text(
        json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten: {run_root / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
