from dataclasses import dataclass, field
import torch
import socket
import threading

from hardware import build_pipeline
from config import SharedConfig, LocalConfig, MSG_CONFIG, MSG_READY, MSG_START, MSG_RESPONSE
from networking.protocol import send_message, read_message
from networking.serialization import to_bytes, serialize_config_query


@dataclass
class UserQuery:
    prompt: str
    model_name: str
    session_id: str
    tokens_to_generate: int
    dtype: torch.dtype = torch.float16

    # No property — dtype is a plain dataclass field.
    # Callers pass torch.float16 / torch.bfloat16 / torch.float32 directly.


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION — the initiator's flow from query to inference
# ═══════════════════════════════════════════════════════════════

def send_query(query, local: LocalConfig, session_manager, daemon_port=65433):
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

    print(f"\n[Orchestrator] Query received: '{query.prompt[:50]}...' "
          f"model={query.model_name} session={query.session_id}")

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
    model_path = local.model_path
    pipeline = build_pipeline(benchmarks, model_path)

    # Step 3: Construct SharedConfig
    shared = SharedConfig(
        port=65432,
        initiator_ip=local.tailscale_ip,
        debug=local.debug,
        pipeline=pipeline,
    )

    # Step 4: Send config to all peers, wait for ready, send start
    peer_ips = [entry["ip"] for entry in pipeline]
    send_shared_config_with_query(query, shared, peer_ips, daemon_port)

    # Step 5: Wait for response from tail
    print("[Orchestrator] Waiting for response...")
    response = receive_response_from_tail(shared.port)

    # Step 6: Update session
    session.add_assistant_message(response, query.model_name)
    session_manager.save_session(session)

    print(f"[Orchestrator] Response received ({len(response)} chars)")
    return response


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
