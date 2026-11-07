"""Run an isolated, repeatable outer/inner n_jobs comparison matrix."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[2]
REPRODUCER = ROOT / "07-midterm" / "reproduce_sklearn_parallel_warning.py"


def read_jsonl(path):
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"event": "invalid_json_record", "raw": line})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=SUITE / "runs")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not REPRODUCER.exists():
        parser.error(f"Reproducer not found: {REPRODUCER}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    env_path = run_root / "environment.json"
    env_command = [
        sys.executable, str(SUITE / "inspect_environment.py"), str(env_path)
    ]
    env_result = subprocess.run(env_command, cwd=ROOT, text=True,
                                capture_output=True)
    (run_root / "environment_inspection.log").write_text(
        env_result.stdout + "\n" + env_result.stderr, encoding="utf-8"
    )
    if env_result.returncode != 0:
        raise SystemExit(
            f"Environment inspection failed; see {run_root}/environment_inspection.log"
        )

    summary_path = run_root / "summary.jsonl"
    configurations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for grid_jobs, forest_jobs in configurations:
        for repetition in range(1, args.repeat + 1):
            attempt = run_root / (
                f"grid-{grid_jobs}_forest-{forest_jobs}"
                f"__repeat-{repetition:02d}"
            )
            capture_dir = attempt / "capture"
            capture_dir.mkdir(parents=True)

            command = [
                sys.executable, "-u", str(REPRODUCER),
                "--grid-n-jobs", str(grid_jobs),
                "--forest-n-jobs", str(forest_jobs),
                "--repeat", "1",
                "--warning-mode", "default",
            ]

            env = os.environ.copy()
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(SUITE) + (
                os.pathsep + old_pythonpath if old_pythonpath else ""
            )
            env["SKLEARN_WARNING_CAPTURE_DIR"] = str(capture_dir)

            started = datetime.now(timezone.utc).isoformat()
            with (attempt / "stdout_stderr.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True
                )

            records = []
            for capture_file in sorted(capture_dir.glob("capture-*.jsonl")):
                records.extend(read_jsonl(capture_file))

            starts = [r for r in records if r.get("event") == "capture_started"]
            warnings_found = [
                r for r in records if r.get("event") == "target_warning"
            ]
            errors = [
                r for r in records
                if r.get("event") in ("invalid_json_record",)
            ]

            if result.returncode != 0:
                status = "failed"
            elif not starts:
                status = "inconclusive_capture_not_verified"
            elif errors:
                status = "inconclusive_capture_records_invalid"
            elif warnings_found:
                status = "warning_observed"
            else:
                status = "no_target_warning_observed_capture_active"

            manifest = {
                "attempt": attempt.name,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "working_directory": str(ROOT),
                "grid_n_jobs": grid_jobs,
                "forest_n_jobs": forest_jobs,
                "repetition": repetition,
                "exit_code": result.returncode,
                "capture_processes_started": len(starts),
                "target_warning_records": len(warnings_found),
                "capture_record_errors": len(errors),
                "status": status,
                "stdout_stderr_log": "stdout_stderr.log",
                "capture_directory": "capture",
            }
            (attempt / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest) + "\n")

            print(
                f"{attempt.name}: {status}; "
                f"warnings={len(warnings_found)}; "
                f"capture processes={len(starts)}; exit={result.returncode}",
                flush=True,
            )

    print(f"\nEvidence directory: {run_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
