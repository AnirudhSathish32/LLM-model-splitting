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

        # generation state (moved from Model + loop locals)
        self.generated_ids = []
        self.full_sequence_ids = None   # set by prepare_request
        self.first_pass_input = None    # set by prepare_request (warm cache slice)
        self.first_pass = True
        self.token_count = 0
        self.cache = None               # DynamicCache, set by prepare_request

        # lifecycle
        self.status = "pending"         # pending → running → done
        self.result = None              # response string, set when done
        self.done_event = threading.Event()
        self.created_at = time.time()
        self.last_stepped_at = 0.0      # 0.0 = never stepped → picked first (round-robin)


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
        self._draining = False

    def submit(self, request):
        """Add a request to the queue. Thread-safe, non-blocking.
        Raises if the Scheduler is draining for a rebuild."""
        if self._draining:
            raise RuntimeError("Scheduler is draining for pipeline rebuild — retry after rebuild")
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
        Pick the next request to step: least recently stepped first.

        Pure round-robin. New requests (last_stepped_at=0.0) get picked
        immediately, then all requests alternate fairly — each gets one
        step per rotation.

        NOTE: an earlier version prioritized decode over prefill over
        pending. In one-step-at-a-time scheduling that starves pending
        requests completely — a running request's decode steps always
        outrank a pending request, so the pending one never starts until
        the running one finishes. Priority tiers only make sense with
        continuous batching, not sequential stepping.
        """
        with self._lock:
            if not self.requests:
                return None
            return min(self.requests, key=lambda r: r.last_stepped_at)

    def active_count(self):
        """Number of in-flight requests (for monitoring)."""
        with self._lock:
            return len(self.requests)

    def drain(self):
        """
        Block until all in-flight requests complete.
        Rejects new submissions while draining.
        Call before pipeline teardown to avoid killing active generation.
        """
        self._draining = True
        count = self.active_count()
        if count > 0:
            print(f"[Scheduler] Draining {count} in-flight requests...")
        while self.active_count() > 0:
            time.sleep(0.1)
        if count > 0:
            print(f"[Scheduler] Drain complete")
        self._draining = False

    def stop(self):
        """Signal the scheduler loop to exit. Call drain() first if requests may be active."""
        self.running = False
        self._has_work.set()
