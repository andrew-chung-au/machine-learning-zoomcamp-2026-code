"""Record environment, input fingerprints, and installed sklearn source details."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[3]
REPRODUCER = ROOT / "07-midterm" / "reproduce_sklearn_parallel_warning.py"
DATASET = ROOT / "07-midterm" / "data" / "training-dataset.arff"
PARALLEL_SOURCE = Path(sklearn.__file__).parent / "utils" / "parallel.py"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    output = Path(sys.argv[1]).resolve()
    source_text = PARALLEL_SOURCE.read_text(encoding="utf-8")
    source_lines = source_text.splitlines()

    # Keep the implementation details relevant to the warning.
    excerpt = []
    for start, end in ((65, 105), (125, 165)):
        excerpt.append({
            "start_line": start,
            "end_line": min(end, len(source_lines)),
            "text": "\n".join(source_lines[start - 1:end]),
        })

    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "context_aware_warnings": getattr(
            sys.flags, "context_aware_warnings", None
        ),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONWARNINGS",
                "PYTHON_CONTEXT_AWARE_WARNINGS",
                "PYTHONPATH",
            )
        },
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "dataset": {
            "path": str(DATASET),
            "exists": DATASET.exists(),
            "sha256": sha256(DATASET) if DATASET.exists() else None,
        },
        "reproducer": {
            "path": str(REPRODUCER),
            "exists": REPRODUCER.exists(),
            "sha256": sha256(REPRODUCER) if REPRODUCER.exists() else None,
        },
        "installed_parallel_source": {
            "path": str(PARALLEL_SOURCE),
            "sha256": sha256(PARALLEL_SOURCE),
            "excerpt": excerpt,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"Environment record: {output}")


if __name__ == "__main__":
    main()
