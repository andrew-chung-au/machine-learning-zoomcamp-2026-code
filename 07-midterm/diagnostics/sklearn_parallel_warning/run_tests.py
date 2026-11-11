"""Run diagnostic-suite regression tests without flooding the terminal."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
ROOT = SUITE_DIR.parents[2]
ROOT_CAUSE_DIR = SUITE_DIR / "root_cause"


def run_step(name, command, *, cwd, env, timeout=60):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"PASS: {name}")
        return True

    print(f"FAIL: {name} (exit {result.returncode})")
    if result.stdout:
        print("--- stdout ---")
        print(result.stdout.rstrip())
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr.rstrip())
    return False


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SUITE_DIR)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    results = []

    with tempfile.TemporaryDirectory(prefix="sklearn-warning-tests-") as tmp:
        env["SKLEARN_WARNING_CAPTURE_DIR"] = str(Path(tmp) / "capture")
        env["SKLEARN_WARNING_TRACE_FILTERS"] = "1"
        Path(env["SKLEARN_WARNING_CAPTURE_DIR"]).mkdir()

        results.append(run_step(
            "synthetic joblib warning capture",
            [sys.executable, str(SUITE_DIR / "test_capture.py")],
            cwd=ROOT,
            env=env,
        ))

        results.append(run_step(
            "filter tracking regression tests",
            [sys.executable, "-m", "unittest", "test_filter_tracking"],
            cwd=SUITE_DIR,
            env=env,
        ))

        results.append(run_step(
            "Python syntax checks",
            [sys.executable, "-m", "py_compile",
             *map(str, sorted(SUITE_DIR.glob("*.py"))),
             *map(str, sorted(ROOT_CAUSE_DIR.rglob("*.py")))],
            cwd=ROOT,
            env=env,
        ))

    # Root-cause checks: scikit-learn's own Parallel/delayed with a fixed
    # thread order (no model fit, no dataset). Each mode checks its own
    # expected outcome and exits non-zero on a mismatch.
    clean = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONWARNINGS", "SKLEARN_WARNING_CAPTURE_DIR",
                 "SKLEARN_WARNING_TRACE_FILTERS", "FILTER_TRACE_DIR"):
        clean.pop(name, None)
    reproducer = [sys.executable, str(ROOT_CAUSE_DIR / "deterministic_reproducer.py")]
    has_context_flag = hasattr(sys.flags, "context_aware_warnings")

    results.append(run_step(
        "deterministic reproducer: warning raised without context-aware warnings",
        reproducer, cwd=ROOT,
        env={**clean, "PYTHON_CONTEXT_AWARE_WARNINGS": "0"},
    ))
    if has_context_flag:
        results.append(run_step(
            "deterministic reproducer: no warning with context-aware warnings",
            reproducer, cwd=ROOT,
            env={**clean, "PYTHON_CONTEXT_AWARE_WARNINGS": "1"},
        ))
    else:
        print("SKIP: context-aware warnings need Python 3.14 or later")
    results.append(run_step(
        "deterministic reproducer: no warning with the snapshot test patch",
        [*reproducer, "--snapshot"], cwd=ROOT,
        env={**clean, "PYTHON_CONTEXT_AWARE_WARNINGS": "0"},
    ))
    results.append(run_step(
        "trace analyzer accounts for every warning in the kept example trace",
        [sys.executable, str(ROOT_CAUSE_DIR / "analyze_trace.py"),
         str(ROOT_CAUSE_DIR / "example-trace"), "--show", "0"],
        cwd=ROOT, env=clean,
    ))

    print()
    passed = sum(results)
    print(f"Summary: {passed}/{len(results)} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
