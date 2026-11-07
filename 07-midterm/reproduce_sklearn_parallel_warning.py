
import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


TARGET_WARNING = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel`"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the scikit-learn parallel-processing warning."
    )
    parser.add_argument(
        "--grid-n-jobs",
        nargs="+",
        type=int,
        choices=[1, -1],
        default=[-1],
        help="GridSearchCV n_jobs values to test.",
    )
    parser.add_argument(
        "--forest-n-jobs",
        nargs="+",
        type=int,
        choices=[1, -1],
        default=[1, -1],
        help="RandomForestClassifier n_jobs values to test.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions for each configuration.",
    )
    parser.add_argument(
        "--warning-mode",
        choices=["default", "error"],
        default="default",
        help=(
            "default: allow warnings; error: configure GridSearchCV "
            "to raise when a fit fails, for diagnostic runs."
        ),
    )
    return parser.parse_args()


def load_data():
    data_path = (
        Path(__file__).resolve().parent
        / "data"
        / "training-dataset.arff"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    raw, _ = arff.loadarff(data_path)
    df = pd.DataFrame(raw)

    # Decode ARFF byte strings and convert feature values to integers.
    df = df.apply(lambda column: column.str.decode("utf-8")).astype(int)
    df.columns = df.columns.str.lower()

    # Convert the original result labels (-1 and 1) to 1 = phishing.
    df["phishing"] = (df["result"] == -1).astype(int)
    df = df.drop(columns=["result"])

    # Use the same stratified 80/20 split as the notebook.
    df_train, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=1,
        stratify=df["phishing"],
    )

    X = df_train.drop(columns=["phishing"])
    y = df_train["phishing"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1,
    )

    return X, y, cv, data_path


def run_search(X, y, cv, grid_n_jobs, forest_n_jobs, run_number, warning_mode):
    grid = {
        "n_estimators": [200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1],
        "max_features": ["sqrt", 0.5],
    }

    print("\n" + "=" * 72, flush=True)
    print(f"Run: {run_number}", flush=True)
    print(f"GridSearchCV n_jobs: {grid_n_jobs}", flush=True)
    print(f"RandomForestClassifier n_jobs: {forest_n_jobs}", flush=True)
    print(f"Warning mode: {warning_mode}", flush=True)
    print("=" * 72, flush=True)

    estimator = RandomForestClassifier(
        random_state=1,
        n_jobs=forest_n_jobs,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=grid_n_jobs,
        error_score="raise" if warning_mode == "error" else np.nan,
    )

    # This recorder captures warnings in this process. It cannot be relied
    # upon to collect warnings emitted separately by joblib worker processes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        try:
            search.fit(X, y)
        except Exception:
            print("RESULT: exception raised during fit.", flush=True)
            traceback.print_exc()
        else:
            print("RESULT: fit completed.", flush=True)
            print(f"Best score: {search.best_score_}", flush=True)
            print(f"Best parameters: {search.best_params_}", flush=True)

    target_warnings = [
        warning
        for warning in caught
        if TARGET_WARNING in str(warning.message)
    ]

    for warning in target_warnings:
        print("TARGET WARNING:", warning.message, flush=True)
        print(
            "Warning location:",
            f"{warning.filename}:{warning.lineno}",
            flush=True,
        )

    if target_warnings:
        print(
            "Target warnings recorded in this process:",
            len(target_warnings),
            flush=True,
        )
    else:
        print(
            "Target warning not recorded in this process.",
            flush=True,
        )
        print(
            "Check terminal output too: the warning may have been "
            "emitted by a worker process.",
            flush=True,
        )


def main():
    args = parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")

    print("Environment", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    print(
        "Context-aware warnings:",
        getattr(sys.flags, "context_aware_warnings", "not available"),
        flush=True,
    )
    print(f"scikit-learn: {sklearn.__version__}", flush=True)
    print(f"SciPy: {scipy.__version__}", flush=True)
    print(f"joblib: {joblib.__version__}", flush=True)
    print(f"NumPy: {np.__version__}", flush=True)
    print(
        "PYTHONWARNINGS:",
        os.environ.get("PYTHONWARNINGS", "(not set)"),
        flush=True,
    )

    X, y, cv, data_path = load_data()

    print(f"Dataset: {data_path}", flush=True)
    print(f"X shape: {X.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print(f"GridSearchCV n_jobs values: {args.grid_n_jobs}", flush=True)
    print(f"Forest n_jobs values: {args.forest_n_jobs}", flush=True)
    print(f"Repetitions: {args.repeat}", flush=True)

    run_number = 0

    for grid_n_jobs in args.grid_n_jobs:
        for forest_n_jobs in args.forest_n_jobs:
            for repetition in range(1, args.repeat + 1):
                run_number += 1
                print(
                    f"\nRepetition {repetition}/{args.repeat}",
                    flush=True,
                )

                run_search(
                    X=X,
                    y=y,
                    cv=cv,
                    grid_n_jobs=grid_n_jobs,
                    forest_n_jobs=forest_n_jobs,
                    run_number=run_number,
                    warning_mode=args.warning_mode,
                )


if __name__ == "__main__":
    main()
