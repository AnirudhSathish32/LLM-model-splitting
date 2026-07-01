from dataclasses import dataclass
import torch
from networking.daemon import discover_and_collect
from hardware import build_pipeline
from config import SharedConfig, LocalConfig, MSG_CONFIG, MSG_READY, MSG_START, MSG_RESPONSE
from networking.protocol import read_message, send_message
import socket
from networking.serialization import to_bytes, serialize_config_query
from networking.protocol import send_message
import threading

@dataclass
class UserQuery:
    prompt: str
    model_name: str
    session_id: str
    tokens_to_generate: int
    dtype: torch.dtype = torch.float16



    @property
    def dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype, torch.float16)
    
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
    5. Send the query to the master (generation begins)
    6. Receive the response from the tail
    7. Update the session and return the response
 
    Parameters:
        query:           Query object from the frontend
        local:           LocalConfig for this machine
        session_id:      Sessioninstance
        daemon_port:     port all daemons listen on
 
    Returns:
        response: str — the model's generated text
    """
    
    session = session_manager.get_or_create(query.session_id)
    session.add_user_message(query.prompt)



    # ── Step 1: Discover peers and collect benchmarks ──────────
 
    print(f"\n[Orchestrator] Query received: '{query.prompt[:50]}...' "
          f"model={query.model_name} session={query.session_id}")
 
    benchmarks, unavailable = discover_and_collect(
        query.model_name,
        daemon_port=daemon_port,
    )
 
    if not benchmarks:
        raise RuntimeError(
            f"No peers have benchmarks for '{query.model_name}'. "
            f"Run benchmark_machine.py on at least one machine first."
        )
 
    if unavailable:
        print(f"[Orchestrator] {len(unavailable)} peers skipped "
              f"(no benchmark for {query.model_name})")
 
    # ── Step 2: Build the pipeline ─────────────────────────────
 
    model_path = local.model_path
    pipeline = build_pipeline(benchmarks, model_path)
 
    # ── Step 3: Construct SharedConfig ─────────────────────────
 
    shared = SharedConfig(
        port=65432,
        initiator_ip=local.tailscale_ip,
        debug=local.debug,
        pipeline=pipeline,
    )
 
    # ── Step 4: Send config to all peers ───────────────────────
    #    Each peer receives the config, looks up its role,
    #    creates an InferencePeer, loads its model slice,
    #    and establishes chain connections.
    #    We wait for all peers to report "ready" before
    #    sending the query.
 
    peer_ips = [entry["ip"] for entry in pipeline]
    master_ip = pipeline[0]["ip"]
 
    send_shared_config_with_query(query, shared, peer_ips, daemon_port)
 
    # ── Step 6: Wait for response from tail ────────────────────
    #    The tail sends the completed response to the initiator_ip
    #    (this machine) when generation finishes.
 
    print("[Orchestrator] Waiting for response...")
    response = receive_response_from_tail(shared.port)
 
    # ── Step 7: Update session ─────────────────────────────────
    
    session.add_assistant_message(response, query.model_name)
    session_manager.save_session(session)
    
    print(f"[Orchestrator] Response received ({len(response)} chars)")
    return response
 
 
# ═══════════════════════════════════════════════════════════════
# CONFIG DISTRIBUTION — send SharedConfig to all peers
# ═══════════════════════════════════════════════════════════════

def send_shared_config_with_query(query, shared, peer_ips, daemon_port=65433):
    """
    Send the SharedConfig to every peer in the pipeline.
    Each peer's daemon receives it and spawns an InferencePeer.
 
    Sends sequentially for simplicity. Could be parallelized with
    threads if handshake latency matters.
    """
    bundle = {
        "shared": to_bytes(shared),
        "query": to_bytes(query)
    }

    payload = serialize_config_query(bundle)
    
    peer_ips = [entry["ip"] for entry in shared.pipeline]
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
                    print(f"{ip}:ready")
                else:
                    sock.close()
                    print(f"{ip} :unexpected response {msg_type}")
        except (ConnectionRefusedError, TimeoutError, ConnectionError) as e:
            print(f"{ip}:failed - {e}")


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

    print(f"[Orchestrator] All {len(connections)} peers ready - sending start signal")
    
    for ip, sock in connections.items():
        send_message(sock, MSG_START)
        sock.close()

    print(f"[Orchestrator] All peers ready - pipeline is live")

# ═══════════════════════════════════════════════════════════════
# RESPONSE RECEIPT — initiator listens for the finished response
# ═══════════════════════════════════════════════════════════════
 
def receive_response_from_tail(inference_port=65432):
    """
    Listen for the tail peer to send the completed response string.
    The tail sends this to initiator_ip when generation finishes.
 
    Uses the inference port (not daemon port) since this is a
    data-plane message — the actual result of generation.
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
 
    response = payload.decode("utf-8")
    return response