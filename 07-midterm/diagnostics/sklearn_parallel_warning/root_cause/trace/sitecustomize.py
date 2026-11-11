"""Load the filter-identity tracer in this process and in joblib workers."""

try:
    import filter_identity_trace

    filter_identity_trace.install()
except Exception as exc:  # the workload must still run if tracing fails
    import sys

    print(f"FILTER_TRACE_INSTALL_ERROR: {exc!r}", file=sys.stderr, flush=True)
