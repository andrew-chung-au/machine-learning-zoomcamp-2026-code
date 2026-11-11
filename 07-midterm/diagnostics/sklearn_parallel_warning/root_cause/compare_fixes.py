"""Compare candidate fixes on the real workload, with the filter-identity tracer on.

Arms (one fresh process tree per attempt, order rotated every round):

    baseline        GridSearchCV and RandomForestClassifier both parallel,
                    PYTHON_CONTEXT_AWARE_WARNINGS=0
    context_aware   as baseline, PYTHON_CONTEXT_AWARE_WARNINGS=1
    forest_single   as baseline, RandomForestClassifier(n_jobs=1): the grid
                    search keeps its worker processes, the forest uses no threads
    snapshot_patch  as baseline, plus the test-only patch that hands tasks a
                    copy of the filter list (candidate upstream change)
    user_filter     as baseline, plus the user's own targeted
                    warnings.filterwarnings("ignore", ...) for this warning
                    (not run by default; select it with --arms). In this arm
                    "raised" (traced) and "shown" (lines in the output) differ
                    if the filter suppresses anything

Every attempt is traced (``trace/filter_identity_trace.py``) and analysed
(``analyze_trace.py``). Two outcomes are recorded per attempt:

    warning     the target warning was raised at least once
    corruption  a task ran with an emptied or shortened filter list, whether
                or not a warning was raised

Evidence goes to ``runs/<UTC timestamp>/`` and is never overwritten:

    environment.json, results.json, summary.jsonl
    attempt-NN-<arm>/manifest.json      status, counts, wall time
    attempt-NN-<arm>/analysis.json      full analyzer output for the attempt
    attempt-NN-<arm>/stdout_stderr.log  raw output of the workload
    attempt-NN-<arm>/trace/*.jsonl.gz   raw traces of affected processes only
                                        (unaffected traces are deleted: about
                                        16 MB per attempt otherwise)

Usage (from anywhere; -1 means "all cores", as in the FAQ example):

    uv run python compare_fixes.py --rounds 15
    uv run python compare_fixes.py --rounds 15 --grid-n-jobs 16 --forest-n-jobs 16

The second form gives a small machine the process and thread layout of a
16-core one. The unit of comparison is the attempt, not the warning count.
"""

import argparse
import gzip
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
ROOT = SUITE.parents[2]
REPRODUCER = ROOT / "07-midterm" / "reproduce_sklearn_parallel_warning.py"
TRACE_DIR = HERE / "trace"

sys.path.insert(0, str(HERE))
import analyze_trace  # noqa: E402

ARMS = {
    "baseline": {"context_aware": 0, "forest_single": False, "snapshot": False},
    "context_aware": {"context_aware": 1, "forest_single": False, "snapshot": False},
    "forest_single": {"context_aware": 0, "forest_single": True, "snapshot": False},
    "snapshot_patch": {"context_aware": 0, "forest_single": False, "snapshot": True},
    # Not a default arm: the user's own targeted "ignore" filter.
    "user_filter": {"context_aware": 0, "forest_single": False, "snapshot": False,
                    "user_filter": True},
}
DEFAULT_ARMS = ["baseline", "context_aware", "forest_single", "snapshot_patch"]
WARNING_LINE = "UserWarning: `sklearn.utils.parallel.delayed` should be used with"


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact p-value for [[a, b], [c, d]] (stdlib only)."""
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2

    def prob(x):
        return (math.comb(row1, x) * math.comb(row2, col1 - x)
                / math.comb(total, col1))

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def run_attempt(index, arm, run_root, grid_n_jobs, forest_n_jobs, trace=True):
    spec = ARMS[arm]
    attempt = run_root / f"attempt-{index:02d}-{arm}"
    trace_dir = attempt / "trace"
    trace_dir.mkdir(parents=True)
    forest = 1 if spec["forest_single"] else forest_n_jobs

    command = [
        sys.executable, "-u", str(REPRODUCER),
        "--grid-n-jobs", str(grid_n_jobs),
        "--forest-n-jobs", str(forest),
        "--repeat", "1",
    ]
    if spec.get("user_filter"):
        command.append("--user-filter")
    env = os.environ.copy()
    env["PYTHON_CONTEXT_AWARE_WARNINGS"] = str(spec["context_aware"])
    env.pop("PYTHONWARNINGS", None)  # a user filter would confound the test
    env.pop("FILTER_TRACE_DIR", None)
    if trace:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(TRACE_DIR) + (os.pathsep + old if old else "")
        env["FILTER_TRACE_DIR"] = str(trace_dir)
        env["FILTER_TRACE_SNAPSHOT"] = "1" if spec["snapshot"] else "0"

    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    log_path = attempt / "stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, text=True)
    wall = time.monotonic() - clock

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    files = analyze_trace.find_traces([trace_dir])
    result, _ = analyze_trace.analyse(files) if files else ({}, {})
    processes = result.get("processes", [])
    flags = sorted({p["context_aware_warnings"] for p in processes}, key=str)
    snapshots = sorted({bool(p["snapshot_patch"]) for p in processes})

    warnings_traced = result.get("target_warnings", 0)
    warnings_in_log = log_text.count(WARNING_LINE)
    corrupted_calls = sum(
        v for k, v in result.get("calls_by_cause", {}).items())
    if completed.returncode != 0:
        status = "failed"
    elif trace and not processes:
        status = "invalid_trace_not_verified"
    elif "FILTER_TRACE_" in log_text:
        status = "invalid_trace_error"
    elif trace and (flags != [spec["context_aware"]]
                    or snapshots != [spec["snapshot"]]):
        status = "invalid_setting_mismatch"
    elif warnings_traced or warnings_in_log:
        status = "warning_observed"
    else:
        status = "no_warning_observed"

    best = [line for line in log_text.splitlines() if line.startswith("Best score")]
    manifest = {
        "attempt": attempt.name, "arm": arm, "spec": spec, "traced": trace,
        "started_utc": started, "wall_seconds": round(wall, 2),
        "command": command, "exit_code": completed.returncode,
        "grid_n_jobs": grid_n_jobs, "forest_n_jobs": forest,
        "processes_traced": len(processes),
        "observed_context_aware_warnings": flags,
        "observed_snapshot_patch": snapshots,
        "target_warnings_traced": warnings_traced,
        "target_warning_lines_in_log": warnings_in_log,
        "warnings_accounted_for": result.get("warnings_accounted_for", 0),
        "calls_by_cause": result.get("calls_by_cause", {}),
        "warnings_by_cause": result.get("warnings_by_cause", {}),
        "decisive_resets": len(result.get("decisive_resets", [])),
        "tasks_with_shortened_list_no_warning": result.get(
            "tasks_with_shortened_list_no_warning", 0),
        "filter_corruption_observed": bool(corrupted_calls) if trace else None,
        "wrapper_tasks_in_threads": result.get("wrapper_tasks_in_threads", 0),
        "best_score_line": best[0] if best else None,
        "status": status,
    }

    # Keep raw traces only for affected processes (compressed); the analysis
    # of every process is in analysis.json.
    affected = {Path(p).name for p in result.get("affected_files", [])}
    kept = []
    for path in files:
        if path.name in affected:
            with path.open("rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            kept.append(path.name + ".gz")
        path.unlink()
    manifest["raw_traces_kept"] = kept
    if result:
        result["affected_files"] = kept
        for call in result.get("affected_calls", []):
            call["file"] = Path(call["file"]).name + ".gz"
        for process in result.get("processes", []):
            process["file"] = Path(process["file"]).name
        (attempt / "analysis.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if not kept and trace_dir.exists():
        trace_dir.rmdir()
    (attempt / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10,
                        help="attempts per arm")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS),
                        default=list(DEFAULT_ARMS))
    parser.add_argument("--grid-n-jobs", type=int, default=-1)
    parser.add_argument("--forest-n-jobs", type=int, default=-1)
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--no-trace", action="store_true",
        help="run without the tracer: the warning is detected from the "
             "workload's output only, corruption is not measured, and the "
             "snapshot arm is not available (use for controls and timing)")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if "baseline" not in args.arms:
        parser.error("the baseline arm is required for the comparison")
    if args.no_trace and "snapshot_patch" in args.arms:
        if args.arms == DEFAULT_ARMS:
            args.arms.remove("snapshot_patch")
        else:
            parser.error("the snapshot arm needs the tracer")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    inspected = subprocess.run(
        [sys.executable, str(SUITE / "inspect_environment.py"),
         str(run_root / "environment.json")],
        cwd=ROOT, text=True, capture_output=True)
    if inspected.returncode != 0:
        (run_root / "environment_inspection.log").write_text(
            inspected.stdout + "\n" + inspected.stderr, encoding="utf-8")
        print("WARNING: environment inspection failed; see the log in "
              f"{run_root}")

    manifests = []
    index = 0
    for round_number in range(args.rounds):
        shift = round_number % len(args.arms)
        order = args.arms[shift:] + args.arms[:shift]
        for arm in order:
            index += 1
            manifest = run_attempt(index, arm, run_root, args.grid_n_jobs,
                                   args.forest_n_jobs, trace=not args.no_trace)
            manifests.append(manifest)
            with (run_root / "summary.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(manifest) + "\n")
            print(f"attempt {index:02d} [{arm:<14}] {manifest['status']:<22} "
                  f"warnings={manifest['target_warnings_traced']:>4} "
                  f"log_lines={manifest['target_warning_lines_in_log']:>4} "
                  f"corruption={str(manifest['filter_corruption_observed']):<5} "
                  f"wall={manifest['wall_seconds']:>6.1f}s", flush=True)

    valid_states = ("warning_observed", "no_warning_observed")
    table = {}
    for arm in args.arms:
        rows = [m for m in manifests if m["arm"] == arm]
        valid = [m for m in rows if m["status"] in valid_states]
        table[arm] = {
            "attempts": len(rows), "valid": len(valid),
            "with_warning": sum(m["status"] == "warning_observed" for m in valid),
            "with_filter_corruption": sum(
                bool(m["filter_corruption_observed"]) for m in valid),
            "target_warnings": sum(m["target_warnings_traced"] for m in valid),
            "target_warning_lines_shown": sum(
                m["target_warning_lines_in_log"] for m in valid),
            "attempts_with_warning_shown": sum(
                bool(m["target_warning_lines_in_log"]) for m in valid),
            "warnings_accounted_for": sum(
                m["warnings_accounted_for"] for m in valid),
            "decisive_resets": sum(m["decisive_resets"] for m in valid),
            "median_wall_seconds": (
                round(statistics.median(m["wall_seconds"] for m in valid), 1)
                if valid else None),
        }
    base = table["baseline"]
    for arm, row in table.items():
        if arm == "baseline" or not (row["valid"] and base["valid"]):
            continue
        row["fisher_p_warning_vs_baseline"] = fisher_exact_two_sided(
            base["with_warning"], base["valid"] - base["with_warning"],
            row["with_warning"], row["valid"] - row["with_warning"])
        row["fisher_p_corruption_vs_baseline"] = fisher_exact_two_sided(
            base["with_filter_corruption"],
            base["valid"] - base["with_filter_corruption"],
            row["with_filter_corruption"],
            row["valid"] - row["with_filter_corruption"])

    print("\n" + "=" * 86)
    print(f"RESULT  (unit = attempt; grid n_jobs={args.grid_n_jobs}, "
          f"forest n_jobs={args.forest_n_jobs} unless forest_single; "
          f"tracer {'off' if args.no_trace else 'on'})")
    print("=" * 86)
    print(f"{'arm':<16}{'valid':>6}{'warned':>8}{'corrupted':>11}"
          f"{'p(warn)':>10}{'p(corrupt)':>12}{'raised':>8}{'shown':>7}"
          f"{'median s':>10}")
    for arm, row in table.items():
        pw = row.get("fisher_p_warning_vs_baseline")
        pc = row.get("fisher_p_corruption_vs_baseline")
        if args.no_trace:
            pc = None
            row["with_filter_corruption"] = "n/a"
            row.pop("fisher_p_corruption_vs_baseline", None)
        print(f"{arm:<16}{row['valid']:>6}{row['with_warning']:>8}"
              f"{row['with_filter_corruption']!s:>11}"
              f"{(f'{pw:.4f}' if pw is not None else '-'):>10}"
              f"{(f'{pc:.4f}' if pc is not None else '-'):>12}"
              f"{(row['target_warnings'] if not args.no_trace else 'n/a')!s:>8}"
              f"{row['target_warning_lines_shown']:>7}"
              f"{row['median_wall_seconds']!s:>10}")
    excluded = [m["attempt"] for m in manifests if m["status"] not in valid_states]
    if excluded:
        print(f"\nExcluded (failed or invalid): {', '.join(excluded)}")
    print("\n'warned': attempts with the target warning. 'corrupted': attempts in "
          "which a task ran\nwith an emptied or shortened filter list (includes "
          "silent cases). p-values: Fisher\nexact, two-sided, against baseline. 'raised': calls of the target\nwarning "
          "(traced). 'shown': warning lines printed by the workload."
          + ("" if args.no_trace else " Wall time includes tracing overhead."))

    (run_root / "results.json").write_text(json.dumps({
        "run_id": run_id, "rounds": args.rounds, "traced": not args.no_trace,
        "grid_n_jobs": args.grid_n_jobs, "forest_n_jobs": args.forest_n_jobs,
        "cpu_count": os.cpu_count(), "arms": table, "excluded": excluded,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvidence directory: {run_root}")


if __name__ == "__main__":
    main()
