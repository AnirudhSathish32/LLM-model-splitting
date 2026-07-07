"""
scheduler.py

Round-robin query interleaving for pipeline-parallel inference.

The Scheduler holds a queue of InFlightRequest objects. Each iteration
it picks the highest-priority request (decode before prefill, then
round-robin by least-recently-stepped), runs one forward step, and
moves to the next.

This lets multiple users see tokens streaming simultaneously instead
of one user blocking everyone until their full response is done.
"""

import threading
import time
from collections import deque


class InFlightRequest:
    """
    Per-request state that persists across steps.

    Everything that currently lives as local variables inside
    run_master_generation moves here: generated_ids, full_sequence_ids,
    first_pass flag, token_count. This lets the Scheduler pause one
    request mid-generation and resume another.
    """

    def __init__(self, query, session, cache_key):
        # identity
        self.query = query
        self.session = session
        self.cache_key = cache_key

        # generation state
        self.generated_ids = []
        self.full_sequence_ids = None   # set by prepare_request
        self.first_pass_input = None    # set by prepare_request (warm cache slice)
        self.first_pass = True
        self.token_count = 0
        self.cache = None

        # lifecycle
        self.status = "pending"         # pending → running → done
        self.result = None              # response string, set when done
        self.done_event = threading.Event()
        self.created_at = time.time()
        self.last_stepped_at = 0.0      # for round-robin within same priority

    @property
    def priority(self):
        """
        Lower number = higher priority.
        Decode steps go before prefill — they're cheaper and keep
        all users' token streams flowing.
        """
        if self.status == "running" and not self.first_pass:
            return 0    # decode step (single token input)
        elif self.status == "running" and self.first_pass:
            return 1    # prefill step (full sequence input)
        else:
            return 2    # pending (not yet started)


class Scheduler:
    """
    Round-robin stepping across multiple in-flight requests.

    The Scheduler runs in its own thread. Daemon threads submit
    requests and block on request.done_event until the Scheduler
    finishes that request's generation.

    Usage:
        scheduler = Scheduler(peer)
        threading.Thread(target=scheduler.run, daemon=True).start()

        # from daemon thread:
        request = InFlightRequest(query, session, cache_key)
        scheduler.submit(request)
        request.done_event.wait()
        response = request.result
    """

    def __init__(self, peer):
        self.peer = peer
        self.requests = deque()
        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self.running = False
        

    def submit(self, request):
        """Add a request to the queue. Thread-safe, non-blocking."""
        with self._lock:
            self.requests.append(request)
        self._has_work.set()

    def run(self):
        """
        Main scheduler loop. Call in a dedicated thread.

        Each iteration:
          1. Pick highest-priority request (decode > prefill > pending)
          2. If pending, prepare it (tokenize, set up first pass)
          3. Run one step (forward + send hidden + receive token)
          4. If done, finalize and signal the waiting daemon thread
          5. If more requests, continue; else wait for new submissions
        """
        self.running = True
        print(f"[Scheduler] Started")

        while self.running:
            # wait for work
            self._has_work.wait(timeout=1.0)

            # process one step at a time until no requests remain
            while self.running:
                request = self._pick_next()
                if request is None:
                    self._has_work.clear()
                    break

                # first step for this request — tokenize and prepare
                if request.status == "pending":
                    request.status = "running"
                    self.peer.prepare_request(request)

                # one forward step
                still_going = self.peer.step_master(request)
                request.last_stepped_at = time.time()

                if not still_going:
                    # generation finished for this request
                    request.status = "done"
                    request.result = self.peer.model.decode(request.generated_ids)
                    request.done_event.set()

                    with self._lock:
                        self.requests.remove(request)

                    print(f"[Scheduler] Request {request.query.session_id} complete "
                          f"({request.token_count} tokens)")

    def _pick_next(self):
        """
        Pick the next request to step.

        Priority order:
          1. Decode steps (status=running, first_pass=False) — cheapest, keeps streams alive
          2. Prefill steps (status=running, first_pass=True) — heavier but in progress
          3. Pending requests (status=pending) — not yet started

        Within the same priority: pick the one least recently stepped
        (round-robin fairness).
        """
        with self._lock:
            if not self.requests:
                return None
            return min(self.requests, key=lambda r: (r.priority, r.last_stepped_at))

    def active_count(self):
        """Number of in-flight requests (for monitoring)."""
        with self._lock:
            return len(self.requests)

    def stop(self):
        """Signal the scheduler loop to exit."""
        self.running = False
        self._has_work.set()