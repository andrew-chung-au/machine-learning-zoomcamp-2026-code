"""Find, in filter-identity traces, the event that empties the stored filter list.

Input: one or more directories containing ``trace-<pid>.jsonl`` (or ``.jsonl.gz``)
files written by ``trace/filter_identity_trace.py`` (searched recursively).

For every process the events are read in the order they were recorded (the
order in which the operations took ``warnings._lock``). The script then

1. groups scikit-learn wrapper tasks that ran in worker threads into inner
   ``Parallel`` calls (consecutive tasks handed the same list object);
2. for each call that produced the target warning, looks for the cause:

   - **emptied during the call**: a ``resetwarnings()`` by a worker thread whose
     target is the very list the tasks were handed, while it still had entries.
     The surrounding events are checked against the proposed interleaving:
     the resetting thread entered ``catch_warnings`` earlier, and before its
     reset another ``catch_warnings`` exit made the handed list global again;
   - **captured already empty**: the list was empty when the first task of the
     call started, i.e. ``Parallel.__call__`` captured a global list that the
     previous call's threads had left behind (a copy taken in the middle of
     another thread's reset, put back as the global list by an out-of-order
     ``catch_warnings`` exit);
   - **captured shortened (silent)**: as above, but the list still had some
     entries, so no warning was raised although filters were lost;

3. prints totals and, with ``--show N``, the event sequence around the first N
   decisive resets (times in microseconds relative to the reset).

``id()`` values are only compared inside one inner call, while the handed list
is referenced by that call's tasks and so cannot have been freed and its
address reused.

Exit status: 0 if every warning is accounted for by one of the two causes,
1 otherwise (or if no trace was found).
"""

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def find_traces(directories):
    return sorted(
        p for d in directories for pattern in ("trace-*.jsonl", "trace-*.jsonl.gz")
        for p in Path(d).rglob(pattern))


def collect(path):
    """Read one process's trace and group thread tasks into inner calls."""
    events = load(path)
    start = next((e for e in events if e["k"] == "START"), {})
    main_tid = None
    # The first catch_warnings entry of a process is in its main thread (the
    # worker threads only exist inside an inner Parallel call).
    for event in events:
        if event["k"] == "E":
            main_tid = event["tid"]
            break

    def new_call(index, handed, handed_len, was_global):
        call = {
            "handed": handed, "first_index": index, "last_index": index,
            "tasks": 0, "tasks_handed_empty": 0, "tasks_handed_shortened": 0,
            "first_task_handed_len": handed_len,
            "handed_was_global_at_first_task": was_global,
            "warnings": 0, "decisive": [],
        }
        calls.append(call)
        return call

    calls = []          # inner Parallel calls (tasks in worker threads), in order
    outer_warnings = 0  # target warnings raised by an outer task (main thread)
    current = None
    for index, event in enumerate(events):
        kind = event["k"]
        if event.get("tid") == main_tid:
            # The main thread only runs between inner calls (it waits during
            # one), so any main-thread event ends the current inner call.
            current = None
            if kind == "W":
                outer_warnings += 1
            continue
        if kind in ("E", "W") and event.get("handed") is not None:
            # The warning is raised just before the task enters catch_warnings,
            # so the first record of a call can be a W.
            if current is None or current["handed"] != event["handed"]:
                current = new_call(index, event["handed"], event["handed_len"],
                                   event.get("handed_is_saved"))
            current["last_index"] = index
            if kind == "W":
                current["warnings"] += 1
            else:
                current["tasks"] += 1
                current["tasks_handed_empty"] += event["handed_len"] == 0
                current["tasks_handed_shortened"] += (
                    0 < event["handed_len"] < current["first_task_handed_len"])
        elif kind == "R" and current is not None:
            if event["list"] == current["handed"] and event["len_before"] > 0:
                current["decisive"].append(index)
                current["last_index"] = index

    return start, events, main_tid, calls, outer_warnings


def classify(path, start, events, main_tid, calls, outer_warnings, full_len):
    """Assign a cause to every inner call that was affected."""
    findings = []
    for call in calls:
        captured_short = 0 < call["first_task_handed_len"] < full_len
        if not (call["warnings"] or call["decisive"] or captured_short):
            continue
        item = dict(call)
        item["pid"] = start.get("pid")
        item["file"] = str(path)
        item["full_len"] = full_len
        if call["decisive"]:
            item["cause"] = "emptied_during_call"
            item["chains"] = [describe_chain(events, i, call) for i in call["decisive"]]
        elif call["first_task_handed_len"] == 0:
            item["cause"] = "captured_already_empty"
            item["capture"] = describe_capture(events, call, main_tid)
        elif captured_short and not call["warnings"]:
            item["cause"] = "captured_shortened_no_warning"
            item["capture"] = describe_capture(events, call, main_tid)
        else:
            item["cause"] = "UNEXPLAINED"
        findings.append(item)

    summary = {
        "file": str(path), "pid": start.get("pid"),
        "context_aware_warnings": start.get("context_aware_warnings"),
        "snapshot_patch": start.get("snapshot_patch"),
        "events": len(events),
        "inner_calls": len(calls),
        "outer_task_warnings": outer_warnings,
        "full_filter_len": full_len,
        "wrapper_tasks_in_threads": sum(c["tasks"] for c in calls),
        "warnings": sum(1 for e in events if e["k"] == "W"),
        "warnings_config_present": sum(
            1 for e in events if e["k"] == "W" and e["config_present"]),
        "warnings_handed_len": dict(collections.Counter(
            e["handed_len"] for e in events if e["k"] == "W")),
        "warnings_global_len": dict(collections.Counter(
            e["current_len"] for e in events if e["k"] == "W")),
        "warnings_handed_is_global": sum(
            1 for e in events if e["k"] == "W" and e["handed"] == e["current"]),
    }
    return summary, findings


def describe_capture(events, call, main_tid):
    """How did the list captured by Parallel.__call__ come to be global?"""
    handed = call["handed"]
    last_worker_exit = None
    for i in range(call["first_index"] - 1, -1, -1):
        e = events[i]
        if e["k"] == "X" and e["tid"] != main_tid:
            last_worker_exit = i
            break
    created = None        # the catch_warnings entry that created this list
    for i in range(call["first_index"] - 1, -1, -1):
        e = events[i]
        if e["k"] == "E" and e["installed"] == handed:
            created = i
            break
    return {
        "left_global_by_worker_thread_exit": (
            last_worker_exit is not None
            and events[last_worker_exit]["restored"] == handed),
        "length_when_left_global": (
            events[last_worker_exit]["restored_len"]
            if last_worker_exit is not None else None),
        "created_by_worker_thread": (
            created is not None and events[created]["tid"] != main_tid),
        "created_as_copy_of_list_with_len": (
            events[created]["saved_len"] if created is not None else None),
    }


def describe_chain(events, reset_index, call):
    """Check the events before a decisive reset against the proposed order."""
    reset = events[reset_index]
    handed = call["handed"]
    tid = reset["tid"]

    own_enter = None      # the resetting thread's catch_warnings entry
    for i in range(reset_index - 1, -1, -1):
        e = events[i]
        if e["k"] == "E" and e["tid"] == tid and e.get("handed") == handed:
            own_enter = i
            break

    made_global = None    # the event that made the handed list global again
    for i in range(reset_index - 1, -1, -1):
        e = events[i]
        if e["k"] == "X" and e["restored"] == handed:
            made_global = i
            break
        if e["k"] == "E" and e["installed"] == handed:
            break

    tasks_before = sum(
        1 for e in events[call["first_index"]:reset_index]
        if e["k"] == "E" and e.get("handed") == handed)
    return {
        "reset_index": reset_index,
        "reset_thread": tid,
        "len_before_reset": reset["len_before"],
        "own_enter_index": own_enter,
        "own_enter_saved_other_list": (
            events[own_enter]["saved"] != handed if own_enter is not None else None),
        "made_global_index": made_global,
        "made_global_by_other_thread": (
            events[made_global]["tid"] != tid if made_global is not None else None),
        "order_enter_then_exit_then_reset": (
            own_enter is not None and made_global is not None
            and own_enter < made_global < reset_index),
        "microseconds_enter_to_reset": (
            (reset["t"] - events[own_enter]["t"]) / 1000
            if own_enter is not None else None),
        "tasks_started_before_reset": tasks_before,
    }


def show_sequence(events, chain, handed):
    reset_index = chain["reset_index"]
    first = min(i for i in (chain["own_enter_index"], chain["made_global_index"],
                            reset_index) if i is not None)
    names = {}

    def name(tid):
        if tid not in names:
            names[tid] = f"T{len(names) + 1}"
        return names[tid]

    def label(list_id):
        return "STORED" if list_id == handed else f"list@{list_id % 100000:05d}"

    t0 = events[reset_index]["t"]
    shown = 0
    for i in range(max(0, first - 2), min(len(events), reset_index + 6)):
        e = events[i]
        when = f"{(e['t'] - t0) / 1000:>10.1f} us"
        who = name(e["tid"])
        if e["k"] == "E":
            task = (f"  [task handed STORED, len {e['handed_len']}]"
                    if e.get("handed") == handed else "")
            text = (f"enter  catch_warnings: saved {label(e['saved'])} "
                    f"(len {e['saved_len']}), global := {label(e['installed'])}{task}")
        elif e["k"] == "X":
            text = (f"exit   catch_warnings: global := {label(e['restored'])} "
                    f"(len {e['restored_len']})")
        elif e["k"] == "R":
            text = (f"resetwarnings(): empties {label(e['list'])} "
                    f"(len {e['len_before']} -> 0)")
        elif e["k"] == "W":
            text = (f"TARGET WARNING: task handed {label(e['handed'])} "
                    f"(len {e['handed_len']}), config present: {e['config_present']}")
        else:
            continue
        marker = "  <== empties the stored list" if i == reset_index else ""
        print(f"    {when}  {who:<4} {text}{marker}")
        shown += 1
    return shown


LABELS = {
    "emptied_during_call": "stored list emptied during the call",
    "captured_already_empty": "captured an already-empty global list",
    "captured_shortened_no_warning": "captured a shortened list (silent)",
    "UNEXPLAINED": "UNEXPLAINED",
}


def analyse(files):
    """Analyse trace files; return (result dict, events of affected files)."""
    collected = [(path, *collect(path)) for path in files]
    # Reference length: the most common length of the list handed to the first
    # task of an inner call, over the whole run.
    lengths = collections.Counter(
        call["first_task_handed_len"] for item in collected for call in item[4])
    full_len = lengths.most_common(1)[0][0] if lengths else 0

    summaries, findings, events_by_file = [], [], {}
    for path, start, events, main_tid, calls, outer_warnings in collected:
        summary, found = classify(
            path, start, events, main_tid, calls, outer_warnings, full_len)
        summaries.append(summary)
        findings.extend(found)
        if found:
            events_by_file[str(path)] = events

    cause_calls = collections.Counter(f["cause"] for f in findings)
    cause_warnings = collections.Counter()
    for f in findings:
        cause_warnings[f["cause"]] += f["warnings"]
    handed_len, global_len = collections.Counter(), collections.Counter()
    for s in summaries:
        handed_len.update(s["warnings_handed_len"])
        global_len.update(s["warnings_global_len"])
    outer = sum(s["outer_task_warnings"] for s in summaries)
    total = sum(s["warnings"] for s in summaries)
    silent_tasks = sum(f["tasks_handed_shortened"] for f in findings) + sum(
        f["tasks"] for f in findings if f["cause"] == "captured_shortened_no_warning")
    result = {
        "trace_files": len(files),
        "processes_with_warning": sum(1 for s in summaries if s["warnings"]),
        "wrapper_tasks_in_threads": sum(
            s["wrapper_tasks_in_threads"] for s in summaries),
        "inner_calls": sum(s["inner_calls"] for s in summaries),
        "usual_filter_list_length": full_len,
        "target_warnings": total,
        "target_warnings_config_present": sum(
            s["warnings_config_present"] for s in summaries),
        "warning_handed_len": {str(k): v for k, v in sorted(handed_len.items())},
        "warning_global_len": {str(k): v for k, v in sorted(global_len.items())},
        "warnings_where_handed_list_was_global": sum(
            s["warnings_handed_is_global"] for s in summaries),
        "calls_by_cause": {c: cause_calls[c] for c in LABELS},
        "warnings_by_cause": {c: cause_warnings[c] for c in LABELS},
        "outer_task_warnings": outer,
        "tasks_with_shortened_list_no_warning": silent_tasks,
        "warnings_accounted_for": total - cause_warnings["UNEXPLAINED"] - outer,
        "decisive_resets": [c for f in findings for c in f.get("chains", [])],
        "captures": [f["capture"] for f in findings if f.get("capture")],
        "affected_calls": [
            {k: v for k, v in f.items() if k != "decisive"} for f in findings],
        "affected_files": sorted(events_by_file),
        "processes": summaries,
    }
    return result, events_by_file


def print_report(result, events_by_file, show):
    r = result
    print(f"Trace files (processes): {r['trace_files']}")
    print(f"Processes with the target warning: {r['processes_with_warning']}")
    print(f"Wrapper tasks run in worker threads: {r['wrapper_tasks_in_threads']}")
    print(f"Inner Parallel calls: {r['inner_calls']} (usual length of the "
          f"filter list handed to tasks: {r['usual_filter_list_length']})")
    print(f"Target warnings: {r['target_warnings']} (configuration present in "
          f"{r['target_warnings_config_present']})")
    print(f"  length of the list the warning task was handed: "
          f"{r['warning_handed_len']}")
    print(f"  length of the global list at that moment:       "
          f"{r['warning_global_len']}")
    print(f"  handed list was itself the global list in "
          f"{r['warnings_where_handed_list_was_global']} warnings")

    print("\nAffected inner Parallel calls, by cause:")
    for cause, text in LABELS.items():
        print(f"  {text:<40} calls={r['calls_by_cause'][cause]:>4}  "
              f"warnings={r['warnings_by_cause'][cause]:>6}")
    print(f"  warnings raised by an outer (process-level) task: "
          f"{r['outer_task_warnings']}")

    captures = r["captures"]
    if captures:
        left = sum(c["left_global_by_worker_thread_exit"] for c in captures)
        made = sum(bool(c["created_by_worker_thread"]) for c in captures)
        copied = collections.Counter(
            c["created_as_copy_of_list_with_len"] for c in captures)
        print(f"\nCalls that captured an empty or shortened list: {len(captures)}")
        print(f"  that list had been left as the global list by a worker "
              f"thread's catch_warnings exit in the previous call: "
              f"{left} of {len(captures)}")
        print(f"  that list had been created by a worker thread's "
              f"catch_warnings entry: {made} of {len(captures)}")
        print(f"  length of the list it was copied from at that moment: "
              f"{dict(copied)}")

    print(f"\nTasks that ran with a shortened, non-empty filter list "
          f"(filters lost, no warning): "
          f"{r['tasks_with_shortened_list_no_warning']}")

    chains = r["decisive_resets"]
    print(f"\nDecisive resets (a worker thread emptied the stored list): "
          f"{len(chains)}")
    if chains:
        ordered = sum(c["order_enter_then_exit_then_reset"] for c in chains)
        other = sum(bool(c["made_global_by_other_thread"]) for c in chains)
        saved_other = sum(bool(c["own_enter_saved_other_list"]) for c in chains)
        gaps = sorted(c["microseconds_enter_to_reset"] for c in chains
                      if c["microseconds_enter_to_reset"] is not None)
        print(f"  resetting thread had entered catch_warnings before the stored "
              f"list became global again, then reset it: {ordered} of {len(chains)}")
        print(f"  stored list was made global again by a different thread's "
              f"exit: {other} of {len(chains)}")
        print(f"  resetting thread's own entry had saved a different list: "
              f"{saved_other} of {len(chains)}")
        if gaps:
            print(f"  time from that thread's entry to its reset (us): "
                  f"min {gaps[0]:.0f}, median {gaps[len(gaps) // 2]:.0f}, "
                  f"max {gaps[-1]:.0f}")

    if r["affected_calls"]:
        print("\nAffected inner calls (tasks in call / tasks that warned / "
              "tasks started before the reset):")
    for f in r["affected_calls"]:
        before = (f["chains"][0]["tasks_started_before_reset"]
                  if f.get("chains") else "-")
        print(f"  pid {f['pid']}: {f['cause']:<30} tasks={f['tasks']:>4} "
              f"warned={f['warnings']:>4} started_before_reset={before}")

    shown = 0
    for f in r["affected_calls"]:
        if shown >= show or not f.get("chains"):
            continue
        print(f"\nEvent sequence, pid {f['pid']} ({Path(f['file']).name}); "
              f"STORED = the list Parallel.__call__ handed to the tasks:")
        show_sequence(events_by_file[f["file"]], f["chains"][0], f["handed"])
        shown += 1

    print(f"\nWarnings accounted for: {r['warnings_accounted_for']} of "
          f"{r['target_warnings']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--show", type=int, default=2,
                        help="number of decisive event sequences to print")
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    files = find_traces(args.directories)
    if not files:
        print("No trace-*.jsonl files found.")
        return 1
    result, events_by_file = analyse(files)
    print_report(result, events_by_file, args.show)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["warnings_accounted_for"] == result["target_warnings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
