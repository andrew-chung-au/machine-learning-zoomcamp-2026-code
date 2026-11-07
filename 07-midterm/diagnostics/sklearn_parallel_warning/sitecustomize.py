"""Automatically install warning capture in Python and joblib worker processes."""

try:
    from capture import install
    install()
except Exception as exc:
    # The runner checks for missing capture_started records.
    import sys
    print(f"WARNING_CAPTURE_INSTALL_ERROR: {exc!r}", file=sys.stderr, flush=True)
