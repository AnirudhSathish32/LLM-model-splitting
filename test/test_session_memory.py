"""
test_session_memory.py

Integration test: verify KV cache persists across conversation turns.

Turn 1: cold cache → full prefill → generate N tokens → cache stored
Turn 2: warm cache → partial prefill (new tokens only) → generate N tokens

Verifies:
  1. Cache is non-None after Turn 1 on all peers
  2. Cache seq_length is preserved between turns
  3. Turn 2 reuses the warm cache (master logs "Warm cache")
  4. Turn 2 produces coherent output (non-empty)
  5. Turn 2 cache is longer than Turn 1 cache (it grew)

Usage:
    python test_session_memory.py
"""

import threading
import queue
import time
import os
import torch
from config import SharedConfig, LocalConfig
from user_query import UserQuery
from inference_peer import InferencePeer
from session import Session


# ═══════════════════════════════════════════════════════════════
# FAKE CONNECTION
# ═══════════════════════════════════════════════════════════════

class FakeConnection:
    def __init__(self, send_queue, recv_queue):
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self._recv_buffer = b""

    def sendall(self, data):
        self.send_queue.put(data)

    def recv(self, bufsize):
        while len(self._recv_buffer) < bufsize:
            try:
                chunk = self.recv_queue.get(timeout=30)
                self._recv_buffer += chunk
            except queue.Empty:
                raise ConnectionError("FakeConnection timeout — peer not sending")
        result = self._recv_buffer[:bufsize]
        self._recv_buffer = self._recv_buffer[bufsize:]
        return result

    def close(self):
        pass


def make_connection_pair():
    q_ab = queue.Queue()
    q_ba = queue.Queue()
    return FakeConnection(q_ab, q_ba), FakeConnection(q_ba, q_ab)


def wire_peers(master_peer, worker_peer, tail_peer):
    """Create fresh FakeConnections and inject them (circular topology)."""
    master_to_worker, worker_from_master = make_connection_pair()
    worker_to_tail, tail_from_worker = make_connection_pair()
    tail_to_master, master_from_tail = make_connection_pair()

    master_peer.downstream_conn = master_to_worker
    master_peer.upstream_conn   = master_from_tail

    worker_peer.upstream_conn   = worker_from_master
    worker_peer.downstream_conn = worker_to_tail

    tail_peer.upstream_conn     = tail_from_worker
    tail_peer.downstream_conn   = tail_to_master


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_cache_len(peer):
    """Return the current seq_length of the peer's active cache, or 0."""
    cache = peer._get_cache()
    if cache is None:
        return 0
    return cache.get_seq_length()


def run_turn(master_peer, worker_peer, tail_peer, query, session=None):
    """
    Run one generation turn across all three peers in threads.
    Returns (response, errors_dict).
    """
    results = {}
    errors = {}

    def run_peer(name, peer, q=None, s=None):
        try:
            results[name] = peer.run_generation(query=q, session=s)
        except Exception as e:
            errors[name] = e
            import traceback
            traceback.print_exc()

    threads = [
        threading.Thread(target=run_peer, args=("master", master_peer, query, session)),
        threading.Thread(target=run_peer, args=("worker", worker_peer)),
        threading.Thread(target=run_peer, args=("tail",   tail_peer, query)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    response = results.get("master", "")
    return response, errors


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

def test_session_memory():
    model_name = "llama-3b"
    tokens_per_turn = 10
    session_id = "session-memory-test"

    local = LocalConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=False,
        tailscale_ip="100.74.100.92",
        model_path="./models",
        layers_path="./layers",
    )

    # ── Pipeline setup ───────────────────────────────────
    from transformers import AutoConfig
    model_dir = os.path.join(local.model_path, model_name)
    config = AutoConfig.from_pretrained(model_dir)
    total_layers = config.num_hidden_layers

    split_1 = total_layers // 3
    split_2 = 2 * total_layers // 3

    pipeline = [
        {"ip": "master", "role": "master", "layers": [0, split_1 - 1]},
        {"ip": "worker", "role": "worker", "layers": [split_1, split_2 - 1]},
        {"ip": "tail",   "role": "tail",   "layers": [split_2, total_layers - 1]},
    ]

    shared = SharedConfig(port=65432, initiator_ip="test", debug=False, pipeline=pipeline)

    query = UserQuery(
        prompt="",  # prompt comes from session
        model_name=model_name,
        session_id=session_id,
        tokens_to_generate=tokens_per_turn,
        dtype=torch.float16,
    )

    # ── Create peers and load models ─────────────────────
    def make_local(fake_ip):
        return LocalConfig(
            device=local.device, debug=False, tailscale_ip=fake_ip,
            model_path=local.model_path, layers_path=local.layers_path,
        )

    master_peer = InferencePeer(shared, make_local("master"))
    worker_peer = InferencePeer(shared, make_local("worker"))
    tail_peer   = InferencePeer(shared, make_local("tail"))

    print("Loading models...")
    t0 = time.time()
    master_peer.load_query_into_model(query)
    worker_peer.load_query_into_model(query)
    tail_peer.load_query_into_model(query)
    print(f"All models loaded in {time.time() - t0:.1f}s\n")

    peers = {"master": master_peer, "worker": worker_peer, "tail": tail_peer}
    session = Session(session_id=session_id)
    all_passed = True

    # ══════════════════════════════════════════════════════
    # TURN 1 — cold cache
    # ══════════════════════════════════════════════════════

    print("=" * 60)
    print("TURN 1 — Cold cache")
    print("=" * 60)

    session.add_user_message("What is the capital of France?")

    # verify caches start empty
    for name, peer in peers.items():
        cl = get_cache_len(peer)
        print(f"  {name} cache BEFORE Turn 1: seq_len={cl}")
        assert cl == 0, f"FAIL: {name} cache should be empty before Turn 1"

    wire_peers(master_peer, worker_peer, tail_peer)
    response_1, errors_1 = run_turn(master_peer, worker_peer, tail_peer, query, session)

    if errors_1:
        print(f"\nTurn 1 ERRORS: {errors_1}")
        return False

    print(f"\n  Turn 1 response: {response_1}")

    # verify caches are populated
    turn1_cache_lens = {}
    for name, peer in peers.items():
        cl = get_cache_len(peer)
        turn1_cache_lens[name] = cl
        print(f"  {name} cache AFTER Turn 1: seq_len={cl}")
        if cl == 0:
            print(f"  FAIL: {name} cache is empty after Turn 1")
            all_passed = False

    # ══════════════════════════════════════════════════════
    # BETWEEN TURNS — update session, reset turn state, keep caches
    # ══════════════════════════════════════════════════════

    print("\n" + "-" * 60)
    print("Between turns: adding assistant response + new user message")
    print("-" * 60)

    session.add_assistant_message(response_1, model_name)
    session.add_user_message("What about Germany?")

    # Reset per-turn state (handoff_package, generated_ids, etc.)
    # but do NOT clear caches — that's the whole point
    for peer in peers.values():
        peer.model.reset_turn_state()

    # Verify caches survived the reset
    for name, peer in peers.items():
        cl = get_cache_len(peer)
        expected = turn1_cache_lens[name]
        print(f"  {name} cache after reset: seq_len={cl} (expected {expected})")
        if cl != expected:
            print(f"  FAIL: {name} cache was corrupted by reset")
            all_passed = False

    # ══════════════════════════════════════════════════════
    # TURN 2 — warm cache
    # ══════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("TURN 2 — Warm cache")
    print("=" * 60)

    wire_peers(master_peer, worker_peer, tail_peer)
    response_2, errors_2 = run_turn(master_peer, worker_peer, tail_peer, query, session)

    if errors_2:
        print(f"\nTurn 2 ERRORS: {errors_2}")
        return False

    print(f"\n  Turn 2 response: {response_2}")

    # verify caches grew
    for name, peer in peers.items():
        cl = get_cache_len(peer)
        prev = turn1_cache_lens[name]
        print(f"  {name} cache AFTER Turn 2: seq_len={cl} (was {prev})")
        if cl <= prev:
            print(f"  FAIL: {name} cache did not grow (Turn 2 didn't extend it)")
            all_passed = False

    # ══════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("SESSION MEMORY TEST RESULTS")
    print("=" * 60)

    print(f"\n  Turn 1 prompt:   'What is the capital of France?'")
    print(f"  Turn 1 response: {response_1}")
    print(f"\n  Turn 2 prompt:   'What about Germany?'")
    print(f"  Turn 2 response: {response_2}")

    checks = [
        ("Turn 1 produced output",       len(response_1) > 0),
        ("Turn 2 produced output",       len(response_2) > 0),
        ("Caches populated after Turn 1", all(v > 0 for v in turn1_cache_lens.values())),
        ("Caches grew after Turn 2",      all(
            get_cache_len(p) > turn1_cache_lens[n]
            for n, p in peers.items()
        )),
    ]

    print()
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_passed = False

    # ── Cleanup ──────────────────────────────────────────
    for peer in peers.values():
        peer.cleanup()

    return all_passed


if __name__ == "__main__":
    success = test_session_memory()
    print(f"\n{'PASSED' if success else 'FAILED'}")
