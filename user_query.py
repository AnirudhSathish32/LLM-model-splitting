from dataclasses import dataclass, field
import torch
import socket
import threading
import time
import os

from hardware import build_pipeline
from config import SharedConfig, LocalConfig, MSG_CONFIG, MSG_READY, MSG_START, MSG_RESPONSE, MSG_QUERY, MSG_QUERY_FAIL
from networking.protocol import send_message, read_message
from networking.serialization import to_bytes, serialize_config_query

import zlib

def _model_port(model_name, base=65432):
    """
    Deterministic port per model so concurrent multi-model pipelines
    don't collide. Uses crc32 (not Python hash(), which is salted
    per-process). Both orchestrators compute the same port for the
    same model independently — no coordination needed.
    """
    offset = zlib.crc32(model_name.encode()) % 500
    return base + offset * 2


@dataclass
class UserQuery:
    prompt: str
    model_name: str
    session_id: str
    tokens_to_generate: int
    dtype: torch.dtype = torch.float16
    messages: list = None 

    # No property — dtype is a plain dataclass field.
    # Callers pass torch.float16 / torch.bfloat16 / torch.float32 directly.


# ═══════════════════════════════════════════════════════════════
# PIPELINE CACHE — per-model, stored after first successful cold query
# ═══════════════════════════════════════════════════════════════

_cached_pipelines = {}   # {model_name: {"shared": ..., "peer_ips": [...], "master_ip": str, "local_only": bool}}

# Local model cache (single-node path, per-model)
_local_models = {}       # {model_name: {"model": nn.Module, "tokenizer": tokenizer}}

# Peer change detection (query-time, rate-limited)
_known_peers = set()
_last_peer_check = 0.0


def _peers_changed_since_check():
    """
    Check if the Tailscale peer set changed since the last check.
    Rate-limited: skips the (slow) Tailscale CLI call if the last
    check was under 60 seconds ago.
    Returns True if peers were added or removed.
    """
    global _last_peer_check, _known_peers

    if time.time() - _last_peer_check < 60.0:
        return False

    from networking.tailscale import get_online_peers
    try:
        current = set(p["ip"] for p in get_online_peers())
    except Exception as e:
        print(f"[Orchestrator] Peer check failed: {e} — proceeding with cached pipeline")
        return False

    _last_peer_check = time.time()

    if not _known_peers:
        _known_peers = current
        return False

    if current != _known_peers:
        added = current - _known_peers
        removed = _known_peers - current
        print(f"[Orchestrator] Peer change detected — "
              f"added: {added or 'none'}, removed: {removed or 'none'}")
        _known_peers = current
        return True

    return False


def clear_pipeline(model_name=None):
    """Clear cached pipeline and local model. If model_name given, clear only that model."""
    global _cached_pipelines, _local_models
    if model_name:
        _cached_pipelines.pop(model_name, None)
        _free_local_model(model_name)
    else:
        _cached_pipelines.clear()
        for name in list(_local_models.keys()):
            _free_local_model(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free_local_model(model_name):
    """Free a single local model from cache."""
    entry = _local_models.pop(model_name, None)
    if entry:
        del entry["model"]
        del entry["tokenizer"]

# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION — the initiator's flow from query to inference
# ═══════════════════════════════════════════════════════════════

def send_query(query, local: LocalConfig, session_manager, daemon_port=65433):
    global _cached_pipelines, _known_peers, _last_peer_check
    """
    Single entry point. Looks up cached pipeline for this model.
    Warm path if found, cold path if not.
    Checks for peer changes at query time (rate-limited to once per 60s).
    """
    from networking.daemon import discover_and_collect

    session = session_manager.get_or_create(query.session_id)
    session.add_user_message(query.prompt)
    query.messages = list(session.messages)

    model_name = query.model_name

    print(f"\n[Orchestrator] Query: '{query.prompt[:50]}...' "
          f"model={model_name} session={query.session_id}")

    # ── Check for peer changes (rate-limited to one check per 60s) ──
    if _peers_changed_since_check():
        print(f"[Orchestrator] Invalidating cached pipelines — cold path will rebuild")
        _cached_pipelines.clear()
    
    # ── Warm path: try reusing existing pipeline for this model ──
    cached = _cached_pipelines.get(model_name)
    if cached is not None:
        try:
            if cached.get("local_only"):
                print(f"[Orchestrator] Using cached local path for {model_name}")
                response = _run_local(query, local, session)
            else:
                response = _try_warm_query(query, cached, daemon_port)

            session.add_assistant_message(response, model_name)
            session_manager.save_session(session)
            print(f"[Orchestrator] Response received via warm path ({len(response)} chars)")
            return response
 
        except Exception as e:
            print(f"[Orchestrator] Warm path failed for {model_name}: {e}")
            print(f"[Orchestrator] Falling back to cold path")
            _cached_pipelines.pop(model_name, None)
 
    # ── Cold path: full discovery + pipeline setup ───────
 
    print(f"[Orchestrator] Running cold path for {model_name}")

    # Step 1: Discover peers and collect benchmarks
    benchmarks, unavailable = discover_and_collect(
        model_name, daemon_port=daemon_port,
    )

    if not benchmarks:
        raise RuntimeError(
            f"No peers have benchmarks for '{model_name}'. "
            f"Run benchmark.py on at least one machine first."
        )

    if unavailable:
        print(f"[Orchestrator] {len(unavailable)} peers skipped "
              f"(no benchmark for {model_name})")

    # Step 2: Build the pipeline
    model_path = os.path.join(local.model_path, model_name)
    pipeline = build_pipeline(benchmarks, model_path, overhead=local.overhead)

    # Refresh peer baseline so the next warm query doesn't re-check immediately
    _known_peers = set(entry["ip"] for entry in pipeline) | {local.tailscale_ip}
    _last_peer_check = time.time()

    # ── Single node: bypass networking entirely ──────────
    if len(pipeline) == 1:
        print(f"[Orchestrator] Single node — running locally, no networking")
        response = _run_local(query, local, session)
 
        _cached_pipelines[model_name] = {"local_only": True}
 
        session.add_assistant_message(response, model_name)
        session_manager.save_session(session)
        print(f"[Orchestrator] Response received via local path ({len(response)} chars)")
        return response
    
    # ── Multi-node: distribute config to peers ───────────

    # Step 3: Construct SharedConfig
    shared = SharedConfig(
        port=_model_port(model_name),
        initiator_ip=local.tailscale_ip,
        debug=local.debug,
        pipeline=pipeline,
    )

    # Step 4+5: Send config to peers, read response from master's
    # control connection. Retry once if the connection drops or the
    # daemon rejects — covers races during concurrent cold starts
    # (the daemon may still be finishing another orchestrator's setup).
    peer_ips = [entry["ip"] for entry in pipeline]
    master_ip = pipeline[0]["ip"]

    response = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            master_conn = send_shared_config_with_query(query, shared, peer_ips, daemon_port)

            print("[Orchestrator] Waiting for response...")
            master_conn.settimeout(300)
            msg_type, payload = read_message(master_conn)
            master_conn.close()

            if msg_type == MSG_RESPONSE:
                response = payload.decode("utf-8")
                break
            elif msg_type == MSG_QUERY_FAIL:
                print(f"[Orchestrator] Daemon rejected query (attempt {attempt+1}/{max_attempts}) "
                      f"— retrying in 2s...")
                time.sleep(2)
            else:
                raise ValueError(f"Expected MSG_RESPONSE, got {msg_type}")

        except (ConnectionError, TimeoutError) as e:
            if attempt < max_attempts - 1:
                print(f"[Orchestrator] Connection dropped (attempt {attempt+1}/{max_attempts}): {e} "
                      f"— retrying in 2s...")
                time.sleep(2)
            else:
                raise

    if response is None:
        raise RuntimeError(f"Query failed after {max_attempts} attempts")

    _cached_pipelines[model_name] = {
        "shared": shared,
        "peer_ips": peer_ips,
        "master_ip": master_ip,
        "local_only": False,
    }

    # Step 6: Update session
    session.add_assistant_message(response, model_name)
    session_manager.save_session(session)

    print(f"[Orchestrator] Response received ({len(response)} chars)")
    return response

# ═══════════════════════════════════════════════════════════════
# LOCAL SINGLE-NODE PATH
# ═══════════════════════════════════════════════════════════════
 
def _run_local(query, local, session):
    """
    Run inference on a single machine — no pipeline splitting, no networking.
    Model is cached per model_name so switching models doesn't reload.
    """
    import os
    global _local_models

    model_name = query.model_name
    model_dir = os.path.join(local.model_path, model_name)

    # Load model on first call for this model, reuse on subsequent calls
    if model_name not in _local_models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Local] Loading full model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=query.dtype
        ).to(local.device)
        model.eval()
        _local_models[model_name] = {"model": model, "tokenizer": tokenizer}
        print(f"[Local] {model_name} loaded on {local.device}")

    model = _local_models[model_name]["model"]
    tokenizer = _local_models[model_name]["tokenizer"]

    # Tokenize full conversation
    prompt_text = tokenizer.apply_chat_template(
        session.messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(local.device)
    input_len = inputs.input_ids.shape[1]

    # Generate
    print(f"[Local] Generating {query.tokens_to_generate} tokens (input: {input_len} tokens)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=query.tokens_to_generate,
            do_sample=False,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
 
    return response

# ═══════════════════════════════════════════════════════════════
# WARM PATH — MSG_QUERY to existing peers
# ═══════════════════════════════════════════════════════════════
 
def _try_warm_query(query, cached, daemon_port):
    """
    Send MSG_QUERY to all peers in the cached pipeline.
    Each peer responds MSG_READY (peer still loaded) or MSG_QUERY_FAIL.
    If all ready: send START, wait for MSG_RESPONSE from master.
    If any fail: raise so caller falls back to cold path.
    """
    peer_ips = cached["peer_ips"]
    master_ip = cached["master_ip"]
 
    payload = to_bytes(query)
 
    connections = {}
    errors = []
    lock = threading.Lock()
 
    def send_and_wait(ip):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((ip, daemon_port))
 
            send_message(sock, MSG_QUERY, payload)
            msg_type, _ = read_message(sock)
 
            with lock:
                if msg_type == MSG_READY:
                    connections[ip] = sock
                    print(f"  {ip}: warm ready")
                elif msg_type == MSG_QUERY_FAIL:
                    sock.close()
                    errors.append(f"{ip}: no loaded peer")
                else:
                    sock.close()
                    errors.append(f"{ip}: unexpected response {msg_type}")
        except (ConnectionRefusedError, TimeoutError, ConnectionError) as e:
            with lock:
                errors.append(f"{ip}: {e}")
 
    print(f"[Orchestrator] Trying warm path to {len(peer_ips)} peers...")
    threads = []
    for ip in peer_ips:
        t = threading.Thread(target=send_and_wait, args=(ip,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
 
    # If any peer failed, clean up and raise
    if errors or len(connections) != len(peer_ips):
        for sock in connections.values():
            sock.close()
        raise RuntimeError(f"Warm path rejected: {'; '.join(errors)}")
 
    # All peers ready — send START, keep master connection open
    for ip, sock in connections.items():
        send_message(sock, MSG_START)
        if ip != master_ip:
            sock.close()
 
    master_conn = connections[master_ip]
    master_conn.settimeout(300)
 
    print(f"[Orchestrator] Warm pipeline running — waiting for response...")
    msg_type, resp_payload = read_message(master_conn)
    master_conn.close()
 
    if msg_type != MSG_RESPONSE:
        raise ValueError(f"Expected MSG_RESPONSE, got {msg_type}")
 
    return resp_payload.decode("utf-8")


# ═══════════════════════════════════════════════════════════════
# CONFIG DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def send_shared_config_with_query(query, shared, peer_ips, daemon_port=65433):
    """
    Send the SharedConfig + Query bundle to every peer in the pipeline.
    Wait for all to report READY, then send START to all.
    Returns the master's control connection (kept open — caller reads
    MSG_RESPONSE from it after generation completes).
    """
    bundle = {
        "shared": to_bytes(shared),
        "query": to_bytes(query),
    }
    payload = serialize_config_query(bundle)

    master_ip = shared.pipeline[0]["ip"]
    connections = {}
    lock = threading.Lock()

    def send_and_wait(ip):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60)
            sock.connect((ip, daemon_port))

            send_message(sock, MSG_CONFIG, payload)
            msg_type, _ = read_message(sock)

            with lock:
                if msg_type == MSG_READY:
                    connections[ip] = sock
                    print(f"  {ip}: ready")
                else:
                    sock.close()
                    print(f"  {ip}: unexpected response {msg_type}")
        except (ConnectionRefusedError, TimeoutError, ConnectionError) as e:
            print(f"  {ip}: failed - {e}")

    print(f"[Orchestrator] Sending config to {len(peer_ips)} peers...")
    threads = []
    for ip in peer_ips:
        t = threading.Thread(target=send_and_wait, args=(ip,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if len(connections) != len(peer_ips):
        failed = [ip for ip in peer_ips if ip not in connections]
        for sock in connections.values():
            sock.close()
        raise RuntimeError(f"Pipeline cannot start. Failed peers: {failed}")

    print(f"[Orchestrator] All {len(connections)} peers ready — sending start signal")
    for ip, sock in connections.items():
        send_message(sock, MSG_START)
        if ip != master_ip:
            sock.close()

    print(f"[Orchestrator] Pipeline is live")
    return connections[master_ip]


# ═══════════════════════════════════════════════════════════════
# RESPONSE RECEIPT
# ═══════════════════════════════════════════════════════════════

def receive_response_from_tail(inference_port=65432):
    """
    Listen for the tail peer to send the completed response string.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", inference_port))
    server.listen(1)

    conn, addr = server.accept()
    msg_type, payload = read_message(conn)

    conn.close()
    server.close()

    if msg_type != MSG_RESPONSE:
        raise ValueError(f"Expected MSG_RESPONSE, got {msg_type}")

    return payload.decode("utf-8")
