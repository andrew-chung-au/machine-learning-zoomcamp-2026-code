"""Regression tests for warning-filter diagnostics."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
TARGET = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel`"
)


class FilterTrackingTests(unittest.TestCase):
    def run_probe(self, operation):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            capture_dir.mkdir()

            env = os.environ.copy()
            env["SKLEARN_WARNING_TRACE_FILTERS"] = "1"
            env["SKLEARN_WARNING_CAPTURE_DIR"] = str(capture_dir)
            env["PYTHONPATH"] = (
                str(SUITE_DIR)
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )

            code = f"""
import warnings
operation = {operation!r}

if operation == "clear":
    warnings.filters.clear()
elif operation == "replace":
    warnings.filters = []

warnings.warn({TARGET!r}, UserWarning)
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"Probe failed. stderr:\\n{result.stderr}",
            )

            records = []
            for file in capture_dir.glob("capture-*.jsonl"):
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            return records

    def test_in_place_clear_is_recorded(self):
        records = self.run_probe("clear")
        self.assertTrue(
            any(r.get("event") == "warning_filters_became_empty" for r in records),
            "Expected an event for clearing the tracked filter list.",
        )

    def test_filter_list_replacement_is_recorded(self):
        records = self.run_probe("replace")
        self.assertTrue(
            any(
                r.get("event") == "warning_filters_replaced_or_untracked"
                for r in records
            ),
            "Expected an event when warnings.filters is replaced.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
