from dataclasses import dataclass, field
import torch
import socket
import threading

from hardware import build_pipeline
from config import SharedConfig, LocalConfig, MSG_CONFIG, MSG_READY, MSG_START, MSG_RESPONSE, MSG_QUERY, MSG_QUERY_FAIL
from networking.protocol import send_message, read_message
from networking.serialization import to_bytes, serialize_config_query


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
# PIPELINE CACHE — stored after first successful cold query
# ═══════════════════════════════════════════════════════════════
 
_cached_pipeline = None  # {"shared": SharedConfig, "peer_ips": [...], "master_ip": str}

# Local model cache (single-node path)
_local_model = None
_local_tokenizer = None
 
 
def clear_pipeline():
    global _cached_pipeline, _local_model, _local_tokenizer
    _cached_pipeline = None
    if _local_model is not None:
        del _local_model
        _local_model = None
    if _local_tokenizer is not None:
        del _local_tokenizer
        _local_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION — the initiator's flow from query to inference
# ═══════════════════════════════════════════════════════════════

def send_query(query, local: LocalConfig, session_manager, daemon_port=65433):
    import os
    global  _cached_pipeline
    """
    Single entry point called by the frontend/API when the user
    sends a prompt. Orchestrates the entire flow:

    1. Discover available peers and collect benchmarks
    2. Build the pipeline (compute the split)
    3. Construct SharedConfig
    4. Distribute config to all peers (they load models + connect)
    5. Receive the response from the tail
    6. Update the session and return the response
    """
    from networking.daemon import discover_and_collect

    session = session_manager.get_or_create(query.session_id)
    session.add_user_message(query.prompt)
    query.messages = list(session.messages)

    print(f"\n[Orchestrator] Query received: '{query.prompt[:50]}...' "
          f"model={query.model_name} session={query.session_id}")
    
    # ── Warm path: try reusing existing pipeline ─────────
    if _cached_pipeline is not None:
        try:
            response = _try_warm_query(query, _cached_pipeline, daemon_port)
 
            session.add_assistant_message(response, query.model_name)
            session_manager.save_session(session)
            print(f"[Orchestrator] Response received via warm path ({len(response)} chars)")
            return response
 
        except Exception as e:
            print(f"[Orchestrator] Warm path failed: {e}")
            print(f"[Orchestrator] Falling back to cold path")
            _cached_pipeline = None
 
    # ── Cold path: full discovery + pipeline setup ───────
 
    print(f"[Orchestrator] Running cold path (discovery + setup)")

    # Step 1: Discover peers and collect benchmarks
    benchmarks, unavailable = discover_and_collect(
        query.model_name, daemon_port=daemon_port,
    )

    if not benchmarks:
        raise RuntimeError(
            f"No peers have benchmarks for '{query.model_name}'. "
            f"Run benchmark.py on at least one machine first."
        )

    if unavailable:
        print(f"[Orchestrator] {len(unavailable)} peers skipped "
              f"(no benchmark for {query.model_name})")

    # Step 2: Build the pipeline
    model_path = os.path.join(local.model_path, query.model_name)
    pipeline = build_pipeline(benchmarks, model_path, overhead=local.overhead)

    # ── Single node: bypass networking entirely ──────────
    if len(pipeline) == 1:
        print(f"[Orchestrator] Single node — running locally, no networking")
        response = _run_local(query, local, session)
 
        _cached_pipeline = {"local_only": True}
 
        session.add_assistant_message(response, query.model_name)
        session_manager.save_session(session)
        print(f"[Orchestrator] Response received via local path ({len(response)} chars)")
        return response
    
    # ── Multi-node: distribute config to peers ───────────

    # Step 3: Construct SharedConfig
    shared = SharedConfig(
        port=65432,
        initiator_ip=local.tailscale_ip,
        debug=local.debug,
        pipeline=pipeline,
    )

    # Step 4: Send config to all peers, wait for ready, send start
    peer_ips = [entry["ip"] for entry in pipeline]
    master_ip = pipeline[0]["ip"]
    send_shared_config_with_query(query, shared, peer_ips, daemon_port)

    # Step 5: Wait for response from tail
    print("[Orchestrator] Waiting for response...")
    response = receive_response_from_tail(shared.port)

    _cached_pipeline = {
        "shared": shared,
        "peer_ips": peer_ips,
        "master_ip": master_ip,
        "local_only": False,
    }

    # Step 6: Update session
    session.add_assistant_message(response, query.model_name)
    session_manager.save_session(session)

    print(f"[Orchestrator] Response received ({len(response)} chars)")
    return response

# ═══════════════════════════════════════════════════════════════
# LOCAL SINGLE-NODE PATH
# ═══════════════════════════════════════════════════════════════
 
def _run_local(query, local, session):
    """
    Run inference on a single machine — no pipeline splitting, no networking.
    Model is cached between calls so subsequent turns skip loading.
    """
    import os
    global _local_model, _local_tokenizer
 
    model_dir = os.path.join(local.model_path, query.model_name)
 
    # Load model on first call, reuse on subsequent calls
    if _local_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Local] Loading full model from {model_dir}...")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=query.dtype
        ).to(local.device)
        _local_model.eval()
        print(f"[Local] Model loaded on {local.device}")
 
    # Tokenize full conversation
    prompt_text = _local_tokenizer.apply_chat_template(
        session.messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _local_tokenizer(prompt_text, return_tensors="pt").to(local.device)
    input_len = inputs.input_ids.shape[1]
 
    # Generate
    print(f"[Local] Generating {query.tokens_to_generate} tokens (input: {input_len} tokens)...")
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=query.tokens_to_generate,
            do_sample=False,
        )
 
    # Decode only the new tokens
    new_tokens = outputs[0][input_len:]
    response = _local_tokenizer.decode(new_tokens, skip_special_tokens=True)
 
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
    """
    bundle = {
        "shared": to_bytes(shared),
        "query": to_bytes(query),
    }
    payload = serialize_config_query(bundle)

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
        sock.close()

    print(f"[Orchestrator] Pipeline is live")


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
