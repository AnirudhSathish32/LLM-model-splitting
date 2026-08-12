"""
logbuffer.py

Captures this process's console output into a bounded, thread-safe ring
buffer so it can be read back over the network and shown in the UI.

Every machine in the pipeline runs its own daemon and prints its own
lines. A node's logs only exist on that node, so the UI pulls from each
one over the existing daemon port (see MSG_LOG_REQ in daemon.py).

Install once, as early as possible:

    from logbuffer import install
    install()

Output still goes to the terminal — this tees, it does not replace.
"""

import io
import re
import sys
import time
import threading
from collections import deque

MAX_LINES = 2000

# Prefixes map to the component that produced the line. Order matters:
# the token pattern is checked first because it is the most specific.
_SOURCES = [
    (re.compile(r"^\[Token \d+\]"), "generation"),
    (re.compile(r"^\[Master\]"),    "generation"),
    (re.compile(r"^\[Tail\]"),      "generation"),
    (re.compile(r"^\[Worker\]"),    "generation"),
    (re.compile(r"^\[Scheduler\]"), "scheduler"),
    (re.compile(r"^\[Daemon\]"),    "daemon"),
    (re.compile(r"^\[Peer\]"),      "daemon"),
    (re.compile(r"^\[Orchestrator\]"), "orchestrator"),
    (re.compile(r"^\[Discovery"),   "orchestrator"),
    (re.compile(r"^\[Local\]"),     "orchestrator"),
    (re.compile(r"^\[Launch\]"),    "system"),
]


def classify(line):
    stripped = line.lstrip()
    for pattern, source in _SOURCES:
        if pattern.match(stripped):
            return source
    return "system"


class LogBuffer:
    """Bounded ring buffer of recent output lines, each with a sequence
    number so readers can poll for only what they have not seen."""

    def __init__(self, maxlen=MAX_LINES):
        self._lines = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, text):
        text = text.rstrip("\n")
        if not text.strip():
            return
        with self._lock:
            self._seq += 1
            self._lines.append({
                "seq": self._seq,
                "t": time.time(),
                "source": classify(text),
                "text": text,
            })

    def read(self, since=0, limit=500):
        """Lines with seq > since, oldest first."""
        with self._lock:
            out = [ln for ln in self._lines if ln["seq"] > since]
        return out[-limit:]

    def latest_seq(self):
        with self._lock:
            return self._seq

    def clear(self):
        with self._lock:
            self._lines.clear()


BUFFER = LogBuffer()


class _Tee(io.TextIOBase):
    """Writes through to the real stream and records complete lines."""

    def __init__(self, stream, buffer):
        self._stream = stream
        self._buffer = buffer
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text):
        try:
            self._stream.write(text)
        except Exception:
            pass

        with self._lock:
            self._partial += text
            if "\n" in self._partial:
                *complete, self._partial = self._partial.split("\n")
                for line in complete:
                    self._buffer.append(line)
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


_installed = False


def install():
    """Tee stdout and stderr into BUFFER. Safe to call more than once."""
    global _installed
    if _installed:
        return BUFFER
    sys.stdout = _Tee(sys.stdout, BUFFER)
    sys.stderr = _Tee(sys.stderr, BUFFER)
    _installed = True
    return BUFFER
