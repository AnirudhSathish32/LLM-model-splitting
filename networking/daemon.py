import os, sys, json, socket, threading, time, torch, io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SharedConfig, LocalConfig,
    MSG_PING, MSG_PONG,
    MSG_BENCHMARK_REQ, MSG_BENCHMARK_RESP, MSG_BENCHMARK_MISS,
    MSG_CONFIG, MSG_READY, MSG_START, MSG_RESPONSE, MSG_QUERY, MSG_QUERY_FAIL
)
from networking.protocol import read_message, send_message
from networking.tailscale import get_online_peers, get_my_ip
from benchmark import load_benchmark
from networking.serialization import from_bytes, to_bytes
from inference_peer import InferencePeer


class Daemon:

    def __init__(self, local_config=None, port=65433):
        self.local = local_config or LocalConfig.load()
        self.port = port
        self.running = False
        self.peer_lock = threading.Lock()
        self.peers = {}              # {model_name: InferencePeer}
        self._peer_last_used = {}    # {model_name: float (timestamp)}

    def start(self):
        """Bind and listen. Blocks forever."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(5)

        server.settimeout(1.0)

        self.running = True

        print(f"[Daemon] Listening on port {self.port}")
        print(f"[Daemon] IP: {self.local.tailscale_ip}")
        print(f"[Daemon] Layers: {self.local.layers_path}")
        print(f"[Daemon] Models: {self.local.model_path}")

        try:
            while self.running:
                try:
                    conn, addr = server.accept()
                    conn.settimeout(None)
                    thread = threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    )
                    thread.start()
                except TimeoutError:
                    continue
        except KeyboardInterrupt:
            print("\n[Daemon] KeyboardInterrupt detected. Shutting down...")
            self.running = False
        finally:
            server.close()

    def _handle_connection(self, conn, addr):
        """Handle one incoming request. Runs in its own thread."""
        try:
            msg_type, payload = read_message(conn)

            if msg_type == MSG_PING:
                self._handle_ping(conn)
            elif msg_type == MSG_BENCHMARK_REQ:
                model_name = payload.decode("utf-8")
                self._handle_benchmark_request(conn, model_name)
            elif msg_type == MSG_CONFIG:
                self._handle_config_query(conn, payload)
            elif msg_type == MSG_QUERY:
                self._handle_query(conn, payload)
            else:
                print(f"[Daemon] Unknown message type {msg_type} from {addr}")

        except ConnectionError as e:
            print(f"[Daemon] Connection error from {addr}: {e}")
        except Exception as e:
            print(f"[Daemon] Error handling {addr}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

    def _handle_ping(self, conn):
        """Respond with availability + our IP."""
        response = json.dumps({
            "available": True,
            "ip": self.local.tailscale_ip,
        }).encode("utf-8")
        send_message(conn, MSG_PONG, response)

    def _handle_benchmark_request(self, conn, model_name):
        """Look up the local benchmark file for the requested model."""
        benchmark_path = f"./benchmark/{model_name}.json"

        if not os.path.exists(benchmark_path):
            print(f"[Daemon] No benchmark for '{model_name}'")
            send_message(conn, MSG_BENCHMARK_MISS, model_name.encode("utf-8"))
            return

        with open(benchmark_path) as f:
            benchmark = json.load(f)

        benchmark["ip"] = self.local.tailscale_ip

        payload = json.dumps(benchmark).encode("utf-8")
        send_message(conn, MSG_BENCHMARK_RESP, payload)
        print(f"[Daemon] Sent benchmark for '{model_name}' to requester")

    def _handle_config_query(self, conn, payload):
        """Receive SharedConfig + Query, create InferencePeer, run generation."""
        from user_query import UserQuery
        from session import Session

        bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        shared = from_bytes(SharedConfig, bundle["shared"])
        query = from_bytes(UserQuery, bundle["query"])
        model_name = query.model_name

        # find our assignment
        my_ip = self.local.tailscale_ip
        my_entry = next(
            (e for e in shared.pipeline if e["ip"] == my_ip),
            None,
        )

        if my_entry is None:
            print(f"[Daemon] WARNING: {my_ip} not in pipeline")
            return
        
        is_master = my_entry["role"] == "master"

        print(f"[Daemon] Role: {my_entry['role']}, "
              f"layers: {my_entry['layers'][0]}..{my_entry['layers'][1]}, "
              f"model: {query.model_name}")

        # tear down existing peer if one is running
        with self.peer_lock:
            if model_name in self.peers:
                self.peers[model_name].cleanup()
                del self.peers[model_name]
                del self._peer_last_used[model_name]

            self._ensure_memory_headroom()

        # create peer, load model
        peer = InferencePeer(shared, self.local)
        peer.load_query_into_model(query)

        with self.peer_lock:
            self.peers[model_name] = peer
            self._peer_last_used[model_name] = time.time()

        # Phase 2: report ready, wait for start signal
        send_message(conn, MSG_READY)
        print(f"[Daemon] Model loaded — waiting for start signal")

        msg_type, _ = read_message(conn)
        if msg_type != MSG_START:
            print(f"[Daemon] Expected MSG_START, got {msg_type}")
            return

        # Phase 3: connect to chain neighbors and run inference
        print(f"[Daemon] Start signal received — connecting chain")
        peer.connect()

        if is_master:
            # Master needs a session to tokenize the conversation
            session = Session(session_id=query.session_id)
            if query.messages:
                session.messages = query.messages
            else:
                session.add_user_message(query.prompt)
 
            response = peer.run_generation(query=query, session=session)
 
            # Send response using the peer's dedicated out-of-band method
            print(f"[Daemon] Generation complete — sending response ({len(response)} chars)")
            send_message(conn, MSG_RESPONSE, response.encode("utf-8"))
        else:
            # Worker/tail — just run, no response to send
            peer.run_generation(query=query)

    def _handle_query(self, conn, payload):
        """
        Handle a warm query — reuse existing peer, model, and connections.
        No model reload, no chain reconnect. Just reset turn state and run.
        """
        from user_query import UserQuery
        from session import Session
        from config import MSG_RESPONSE
 
        query = from_bytes(UserQuery, payload)
        model_name = query.model_name
 
        with self.peer_lock:
            peer = self.peers.get(model_name)
            if peer is not None:
                self._peer_last_used[model_name] = time.time()
 
        if peer is None or peer.model is None:
            print(f"[Daemon] No peer loaded — rejecting MSG_QUERY")
            send_message(conn, MSG_QUERY_FAIL)
            return
 
        # Reset turn state but keep model, connections, and caches
        peer.model.reset_turn_state()
 
        # Update cache key for this query's session
        cache_key = (query.session_id, query.model_name)
        if cache_key not in peer.caches:
            peer.caches[cache_key] = None
        peer._active_cache_key = cache_key
 
        is_master = peer.is_master
 
        # Synchronize: report ready, wait for start
        send_message(conn, MSG_READY)
        print(f"[Daemon] Warm query ready (reusing peer, role={peer.role})")
 
        msg_type, _ = read_message(conn)
        if msg_type != MSG_START:
            print(f"[Daemon] Expected MSG_START, got {msg_type}")
            return
 
        print(f"[Daemon] Warm query start — running generation")
 
        if is_master:
            session = Session(session_id=query.session_id)
            if query.messages:
                session.messages = query.messages
            else:
                session.add_user_message(query.prompt)
 
            response = peer.run_generation(query=query, session=session)
 
            print(f"[Daemon] Warm query complete — sending MSG_RESPONSE ({len(response)} chars)")
            send_message(conn, MSG_RESPONSE, response.encode("utf-8"))
            print(f"[Daemon] MSG_RESPONSE sent")
        else:
            peer.run_generation(query=query)
            print(f"[Daemon] Warm query complete (tail/worker)")

    def _get_memory_status(self):
        """Returns (total_bytes, free_bytes) for the compute device."""
        import psutil
        device = self.local.device
        if device.startswith("cuda"):
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            free = total - torch.cuda.memory_allocated(0)
        else:
            mem = psutil.virtual_memory()
            total = mem.total
            free = mem.available
        return total, free
 
    def _ensure_memory_headroom(self):
        """Evict LRU models until free memory >= overhead * total."""
        overhead = self.local.overhead
        total, free = self._get_memory_status()
        required_free = int(total * overhead)
 
        while free < required_free and self._peer_last_used:
            total_gb = total / 1e9
            free_gb = free / 1e9
            req_gb = required_free / 1e9
            print(f"[Daemon] Memory tight: {free_gb:.1f}GB free < {req_gb:.1f}GB required "
                  f"({overhead*100:.0f}% of {total_gb:.1f}GB)")
            self._evict_lru_peer()
            total, free = self._get_memory_status()
 
    def _evict_lru_peer(self):
        """Evict the least recently used model to free memory."""
        if not self._peer_last_used:
            return
        oldest = min(self._peer_last_used, key=self._peer_last_used.get)
        print(f"[Daemon] Evicting LRU model '{oldest}'")
        self.peers[oldest].cleanup()
        del self.peers[oldest]
        del self._peer_last_used[oldest]


# ================================================================
# INITIATOR-SIDE: discovery and benchmark collection
# ================================================================

def ping_peer(ip, port=65433, timeout=5):
    """Ping a single peer's daemon. Returns availability dict or None."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        send_message(sock, MSG_PING)
        msg_type, payload = read_message(sock)
        sock.close()
        if msg_type == MSG_PONG:
            return json.loads(payload.decode("utf-8"))
        return None
    except (ConnectionRefusedError, TimeoutError, ConnectionError):
        return None


def request_benchmark(ip, model_name, port=65433, timeout=10):
    """Request a specific model's benchmark from a single peer."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        send_message(sock, MSG_BENCHMARK_REQ, model_name.encode("utf-8"))
        msg_type, payload = read_message(sock)
        sock.close()

        if msg_type == MSG_BENCHMARK_RESP:
            return json.loads(payload.decode("utf-8"))
        return None
    except (ConnectionRefusedError, TimeoutError, ConnectionError):
        return None


def discover_and_collect(model_name, daemon_port=65433):
    """
    Full discovery + benchmark collection flow.
    Called by the initiator when a query arrives.
    """
    my_ip = get_my_ip()
    peers = get_online_peers()

    print(f"\n[Discovery] Found {len(peers)} peers on tailnet")

    available = []
    for peer in peers:
        ip = peer["ip"]
        if ip == my_ip:
            continue

        pong = ping_peer(ip, port=daemon_port)
        if pong and pong.get("available"):
            available.append(ip)
            print(f"  {ip} ({peer.get('hostname', '?')}): available")
        else:
            print(f"  {ip} ({peer.get('hostname', '?')}): unavailable")

    print(f"\n[Discovery] {len(available)} peers available, requesting benchmarks...")
    benchmarks = []
    unavailable = []

    for ip in available:
        bench = request_benchmark(ip, model_name, port=daemon_port)
        if bench:
            benchmarks.append(bench)
            print(f"  {ip}: benchmark received "
                  f"({bench.get('layer_time_s', 0)*1000:.2f} ms/layer, "
                  f"{bench.get('gpu_name') or 'CPU'})")
        else:
            unavailable.append(ip)
            print(f"  {ip}: no benchmark for '{model_name}'")

    # include our own benchmark
    my_bench = load_benchmark(model_name)
    if my_bench:
        my_bench["ip"] = my_ip
        benchmarks.append(my_bench)
        print(f"  {my_ip} (self): benchmark loaded locally")

    print(f"\n[Discovery] Collected {len(benchmarks)} benchmarks, "
          f"{len(unavailable)} peers missing benchmark")

    return benchmarks, unavailable


if __name__ == "__main__":
    local = LocalConfig.load()

    if not local.tailscale_ip:
        print("ERROR: tailscale_ip not set. Run LocalConfig setup or set TAILSCALE_IP env var.")
        sys.exit(1)

    daemon = Daemon(local_config=local)
    daemon.start()
