"""
test_happy_path.py

Integration test: 3 peers (master, worker, tail) running in threads
on the same machine, connected via queues instead of TCP.

Usage:
    python test_happy_path.py

Requirements:
    - Layer files exist at local.layers_dir/<model_name>/
    - Model files exist at local.model_dir/<model_name>/
    - Enough RAM/VRAM to hold 3 slices simultaneously
"""

import threading
import queue
import time
import torch
from config import SharedConfig, LocalConfig
from query import Query
from inference_peer import InferencePeer
from networking.protocol import send_message, read_message


# ═══════════════════════════════════════════════════════════════
# FAKE CONNECTION — replaces TCP with thread-safe queues
# ═══════════════════════════════════════════════════════════════

class FakeConnection:
    """
    Drop-in replacement for a TCP socket. Two FakeConnections are
    linked by a shared queue — sendall on one side puts bytes into
    the queue, recv on the other side pulls them out. Thread-safe
    because queue.Queue handles locking internally.
    """

    def __init__(self, send_queue, recv_queue):
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self._recv_buffer = b""

    def sendall(self, data):
        self.send_queue.put(data)

    def recv(self, bufsize):
        # drain the queue into the buffer if empty
        while len(self._recv_buffer) < bufsize:
            try:
                chunk = self.recv_queue.get(timeout=30)
                self._recv_buffer += chunk
            except queue.Empty:
                raise ConnectionError("FakeConnection timeout — peer not sending")

        # return requested bytes, keep the rest
        result = self._recv_buffer[:bufsize]
        self._recv_buffer = self._recv_buffer[bufsize:]
        return result

    def close(self):
        pass


def make_connection_pair():
    """
    Create two linked FakeConnections. What one sends, the other receives.

    Returns (conn_a, conn_b) — a sends to b, b sends to a.
    """
    q_a_to_b = queue.Queue()
    q_b_to_a = queue.Queue()

    conn_a = FakeConnection(send_queue=q_a_to_b, recv_queue=q_b_to_a)
    conn_b = FakeConnection(send_queue=q_b_to_a, recv_queue=q_a_to_b)

    return conn_a, conn_b


# ═══════════════════════════════════════════════════════════════
# TEST SETUP
# ═══════════════════════════════════════════════════════════════

def test_happy_path():
    model_name = "llama-3b"     # use your smaller model for testing
    prompt = "hello there"
    tokens = 10

    local = LocalConfig.load()

    # ── Build a 3-node pipeline manually ─────────────────
    # In production, build_pipeline computes this from benchmarks.
    # For testing, we hardcode a split.

    from transformers import AutoConfig
    model_path = f"{local.model_dir}/{model_name}"
    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers

    # split roughly into thirds
    split_1 = total_layers // 3
    split_2 = 2 * total_layers // 3

    pipeline = [
        {"ip": "master",  "role": "master", "layers": [0, split_1 - 1]},
        {"ip": "worker",  "role": "worker", "layers": [split_1, split_2 - 1]},
        {"ip": "tail",    "role": "tail",   "layers": [split_2, total_layers - 1]},
    ]

    print(f"Pipeline: {total_layers} layers")
    for entry in pipeline:
        count = entry["layers"][1] - entry["layers"][0] + 1
        print(f"  {entry['role']}: layers {entry['layers'][0]}..{entry['layers'][1]} ({count} layers)")

    shared = SharedConfig(
        port=65432,
        initiator_ip="test",
        debug=False,
        pipeline=pipeline,
    )

    query = Query(
        prompt=prompt,
        model_name=model_name,
        session_id="test-session",
        tokens_to_generate=tokens,
        dtype=torch.float16,
    )

    # ── Create fake connections ──────────────────────────
    # master ←→ worker ←→ tail
    master_to_worker, worker_from_master = make_connection_pair()
    worker_to_tail, tail_from_worker = make_connection_pair()

    # tail sends tokens back to master (reverse channel)
    tail_to_master, master_from_tail = make_connection_pair()

    # ── Create peers with overridden IPs ─────────────────
    # Override tailscale_ip on local config so each peer
    # finds itself in the pipeline by its fake IP

    def make_local(fake_ip):
        l = LocalConfig.load()
        l.tailscale_ip = fake_ip
        return l

    master_peer = InferencePeer(shared, make_local("master"))
    worker_peer = InferencePeer(shared, make_local("worker"))
    tail_peer   = InferencePeer(shared, make_local("tail"))

    # ── Load models ──────────────────────────────────────
    print("\nLoading models...")
    t0 = time.time()

    master_peer.load_query(query)
    worker_peer.load_query(query)
    tail_peer.load_query(query)

    print(f"All models loaded in {time.time() - t0:.1f}s")

    # ── Inject fake connections ──────────────────────────
    # Bypass peer.connect() — inject the queue-backed connections directly

    master_peer.downstream_conn = master_to_worker
    master_peer.upstream_conn   = master_from_tail    # receives tokens from tail

    worker_peer.upstream_conn   = worker_from_master
    worker_peer.downstream_conn = worker_to_tail

    tail_peer.upstream_conn     = tail_from_worker
    tail_peer.downstream_conn   = tail_to_master      # sends tokens to master

    # ── Run all three peers in threads ───────────────────
    results = {}
    errors = {}

    def run_peer(name, peer, query, session=None):
        try:
            result = peer.run_generation(query, session)
            results[name] = result
        except Exception as e:
            errors[name] = e
            import traceback
            traceback.print_exc()

    from session import Session
    session = Session(session_id="test-session")
    session.add_user_message(prompt)

    print(f"\nStarting generation: '{prompt}' → {tokens} tokens\n")
    t0 = time.time()

    master_thread = threading.Thread(
        target=run_peer,
        args=("master", master_peer, query, session),
    )
    worker_thread = threading.Thread(
        target=run_peer,
        args=("worker", worker_peer, query),
    )
    tail_thread = threading.Thread(
        target=run_peer,
        args=("tail", tail_peer, query),
    )

    master_thread.start()
    worker_thread.start()
    tail_thread.start()

    master_thread.join(timeout=120)
    worker_thread.join(timeout=120)
    tail_thread.join(timeout=120)

    gen_time = time.time() - t0

    # ── Report results ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")

    if errors:
        print(f"\nERRORS:")
        for name, e in errors.items():
            print(f"  {name}: {e}")

    if "master" in results:
        print(f"\nResponse: {results['master']}")
    elif "tail" in results:
        print(f"\nResponse: {results['tail']}")

    print(f"\nGeneration time: {gen_time:.2f}s")
    print(f"Tokens: {tokens}")
    print(f"Per token: {gen_time/tokens*1000:.0f}ms")
    print(f"{'='*60}")

    # ── Cleanup ──────────────────────────────────────────
    master_peer.model.unload()
    worker_peer.model.unload()
    tail_peer.model.unload()

    return len(errors) == 0


if __name__ == "__main__":
    success = test_happy_path()
    print(f"\n{'PASSED' if success else 'FAILED'}")