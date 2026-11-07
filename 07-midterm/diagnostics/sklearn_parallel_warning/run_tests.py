"""Run diagnostic-suite regression tests without flooding the terminal."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
ROOT = SUITE_DIR.parents[2]


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
             *map(str, sorted(SUITE_DIR.glob("*.py")))],
            cwd=ROOT,
            env=env,
        ))

    print()
    passed = sum(results)
    print(f"Summary: {passed}/{len(results)} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
