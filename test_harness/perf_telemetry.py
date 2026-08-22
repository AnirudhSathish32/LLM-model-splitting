"""
telemetry.py

Per-token latency records for the measurement experiments.

Disabled unless enabled explicitly, so normal inference pays nothing.
The master is the only node that records: it is the one point that sees
both its own compute and the full round trip.

What is measured directly:
    master_compute_ms  time inside the master's own forward pass
    roundtrip_ms       from the moment the hidden state is handed to the
                       socket until a token comes back — network in both
                       directions plus every downstream stage's compute

What must be derived (and labelled as such in any writeup):
    downstream compute, estimated from each node's benchmarked ms/layer
    network time = roundtrip - estimated downstream compute
"""

import csv
import os
import threading
import time

_enabled = False
_records = []
_lock = threading.Lock()
_run_meta = {}


def enable(**meta):
    """Start collecting. Any keyword arguments are attached to every record."""
    global _enabled, _run_meta
    _enabled = True
    _run_meta = dict(meta)


def disable():
    global _enabled
    _enabled = False


def is_enabled():
    return _enabled


def record(**fields):
    """Store one token's timing. No-op when disabled."""
    if not _enabled:
        return
    row = dict(_run_meta)
    row.update(fields)
    row["wall_time"] = time.time()
    with _lock:
        _records.append(row)


def clear():
    with _lock:
        _records.clear()


def rows():
    with _lock:
        return list(_records)


def write_csv(path):
    """Append records to a CSV, writing a header if the file is new."""
    with _lock:
        data = list(_records)
    if not data:
        print(f"[Telemetry] nothing to write to {path}")
        return 0

    keys = []
    for row in data:
        for k in row:
            if k not in keys:
                keys.append(k)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if is_new:
            writer.writeheader()
        writer.writerows(data)

    print(f"[Telemetry] wrote {len(data)} rows to {path}")
    return len(data)