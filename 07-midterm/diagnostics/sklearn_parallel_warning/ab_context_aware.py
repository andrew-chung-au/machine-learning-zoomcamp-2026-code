"""A/B test: PYTHON_CONTEXT_AWARE_WARNINGS off vs on, real workload.

Runs only the configuration that has produced the warning (GridSearchCV
n_jobs=-1 with RandomForestClassifier n_jobs=-1), alternating the two settings
(off, on, off, on, ...) so slow drift on the machine affects both equally.

Every attempt is a fresh Python process, with the capture hook active in it and
in its joblib workers. Evidence goes to ``ab-runs/<UTC timestamp>/`` (never
overwritten):

    environment.json            package versions, hashes (inspect_environment.py)
    summary.jsonl               one manifest per attempt
    results.json                counts and the statistical comparison
    attempt-NN-off|on/
        manifest.json
        stdout_stderr.log       raw terminal output (warning lines go here)
        capture/capture-<pid>.jsonl   one file per process

Run from anywhere:

    uv run python 07-midterm/diagnostics/sklearn_parallel_warning/ab_context_aware.py --pairs 10

What the result can and cannot show
-----------------------------------
- The unit is the *attempt* (one fresh process tree), not the warning count.
  One attempt may log 200 warnings; that is one event.
- Fisher's exact test (two-sided) is printed for the 2x2 table. With few pairs
  it has little power: a clean "on" arm after 3 attempts proves nothing.
- A difference supports, but does not prove, that context-aware warnings
  prevent the problem. It says nothing about the cause.
- An attempt is only counted if the child process really reported the
  intended ``context_aware_warnings`` value; otherwise it is "invalid".
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[2]
DEFAULT_REPRODUCER = ROOT / "07-midterm" / "reproduce_sklearn_parallel_warning.py"


def read_jsonl(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"event": "invalid_json_record", "raw": line})
    return records


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


def run_attempt(index, arm, run_root, reproducer, extra_args):
    expected_flag = 1 if arm == "on" else 0
    attempt = run_root / f"attempt-{index:02d}-{arm}"
    capture_dir = attempt / "capture"
    capture_dir.mkdir(parents=True)

    command = [
        sys.executable, "-u", str(reproducer),
        "--grid-n-jobs", "-1",
        "--forest-n-jobs", "-1",
        "--repeat", "1",
        "--warning-mode", "default",
        *extra_args,
    ]

    env = os.environ.copy()
    env["PYTHON_CONTEXT_AWARE_WARNINGS"] = str(expected_flag)  # explicit both ways
    env.pop("PYTHONWARNINGS", None)  # a user filter would confound the test
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SUITE) + (os.pathsep + old if old else "")
    env["SKLEARN_WARNING_CAPTURE_DIR"] = str(capture_dir)

    started = datetime.now(timezone.utc).isoformat()
    with (attempt / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True)

    records = []
    for capture_file in sorted(capture_dir.glob("capture-*.jsonl")):
        records.extend(read_jsonl(capture_file))

    starts = [r for r in records if r.get("event") == "capture_started"]
    hits = [r for r in records if r.get("event") == "target_warning"]
    bad = [r for r in records if r.get("event") == "invalid_json_record"]
    flags = sorted({r.get("context_aware_warnings") for r in starts}, key=str)

    if result.returncode != 0:
        status = "failed"
    elif not starts:
        status = "invalid_capture_not_verified"
    elif bad:
        status = "invalid_capture_records"
    elif flags != [expected_flag]:
        status = "invalid_context_flag_mismatch"
    elif hits:
        status = "warning_observed"
    else:
        status = "no_warning_observed"

    manifest = {
        "attempt": attempt.name,
        "arm": arm,
        "expected_context_aware_warnings": expected_flag,
        "observed_context_aware_warnings": flags,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": result.returncode,
        "capture_processes_started": len(starts),
        "target_warning_records": len(hits),
        "warning_pids": sorted({r.get("pid") for r in hits}),
        "passed_filter_lengths": sorted(
            {r.get("passed_warning_filters_length") for r in hits}, key=str),
        "status": status,
    }
    (attempt / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=10,
                        help="number of off/on pairs (2*pairs attempts total)")
    parser.add_argument("--output-root", type=Path, default=SUITE / "ab-runs")
    parser.add_argument("--reproducer", type=Path, default=DEFAULT_REPRODUCER,
                        help="override the workload script (used for self-tests)")
    args, extra = parser.parse_known_args()
    args.reproducer = args.reproducer.resolve()  # children run from the repo root

    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if not args.reproducer.exists():
        parser.error(f"Reproducer not found: {args.reproducer}")
    if getattr(sys.flags, "context_aware_warnings", None) is None:
        print("NOTE: this Python has no sys.flags.context_aware_warnings "
              "(needs 3.14+). The 'on' arm cannot be tested; aborting.")
        raise SystemExit(2)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    env_result = subprocess.run(
        [sys.executable, str(SUITE / "inspect_environment.py"),
         str(run_root / "environment.json")],
        cwd=ROOT, text=True, capture_output=True)
    (run_root / "environment_inspection.log").write_text(
        env_result.stdout + "\n" + env_result.stderr, encoding="utf-8")
    if env_result.returncode != 0:
        print("WARNING: environment inspection failed; see "
              f"{run_root}/environment_inspection.log")

    manifests = []
    index = 0
    for pair in range(1, args.pairs + 1):
        # Alternate which arm goes first so order effects cancel too.
        order = ("off", "on") if pair % 2 else ("on", "off")
        for arm in order:
            index += 1
            manifest = run_attempt(index, arm, run_root, args.reproducer, extra)
            manifests.append(manifest)
            with (run_root / "summary.jsonl").open("a", encoding="utf-8") as s:
                s.write(json.dumps(manifest) + "\n")
            print(f"attempt {index:02d} [{arm:>3}]: {manifest['status']}; "
                  f"warning records={manifest['target_warning_records']}; "
                  f"processes={manifest['capture_processes_started']}",
                  flush=True)

    def tally(arm):
        rows = [m for m in manifests if m["arm"] == arm]
        valid = [m for m in rows if m["status"] in
                 ("warning_observed", "no_warning_observed")]
        warned = sum(m["status"] == "warning_observed" for m in valid)
        return len(rows), len(valid), warned

    off_total, off_valid, off_warned = tally("off")
    on_total, on_valid, on_warned = tally("on")

    print("\n" + "=" * 64)
    print("A/B RESULT  (unit = attempt; GridSearchCV -1, forest -1)")
    print("=" * 64)
    print(f"{'context-aware':<16}{'attempts':>9}{'valid':>7}"
          f"{'with warning':>14}{'rate':>8}")
    for name, total, valid, warned in (
            ("off (flag=0)", off_total, off_valid, off_warned),
            ("on  (flag=1)", on_total, on_valid, on_warned)):
        rate = f"{warned / valid:.0%}" if valid else "n/a"
        print(f"{name:<16}{total:>9}{valid:>7}{warned:>14}{rate:>8}")

    p_value = None
    if off_valid and on_valid:
        p_value = fisher_exact_two_sided(
            off_warned, off_valid - off_warned,
            on_warned, on_valid - on_warned)
        print(f"\nFisher exact (two-sided) p = {p_value:.4f}")

    invalid = [m["attempt"] for m in manifests
               if m["status"] not in ("warning_observed", "no_warning_observed")]
    if invalid:
        print(f"\nExcluded (failed or invalid): {', '.join(invalid)}")

    print("\nReading this result:")
    if off_warned == 0:
        print("- The 'off' arm never warned, so this run cannot test the "
              "workaround. Repeat with more pairs.")
    elif on_warned == 0 and off_warned > 0:
        print("- Warning seen only with the flag off. Consistent with the "
              "workaround helping; that is not proof, and small counts "
              "are fragile.")
    else:
        print("- Warnings occurred with the flag on. The workaround did not "
              "prevent them in this run; that contradicts the explanation.")
    print("- Do not treat a high p-value as 'no difference'; with few attempts "
          "the test has little power.")

    results = {
        "run_id": run_id,
        "off": {"attempts": off_total, "valid": off_valid, "with_warning": off_warned},
        "on": {"attempts": on_total, "valid": on_valid, "with_warning": on_warned},
        "fisher_exact_two_sided_p": p_value,
        "excluded": invalid,
    }
    (run_root / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvidence directory: {run_root}")


if __name__ == "__main__":
    main()
