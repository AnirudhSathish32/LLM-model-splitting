import os, sys, json, socket, threading, time, torch, io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MSG_TOKEN_STREAM,
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
        self.schedulers = {}         # {model_name: Scheduler} (master only)
        self._gen_threads = {}       # {model_name: Thread} (worker/tail long-running loops)
        self._creating = set()       # model_names currently being created (guards race)

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
        from scheduler import Scheduler, InFlightRequest

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

        print(f"[Daemon] Role: {my_entry['role']}, "
              f"layers: {my_entry['layers'][0]}..{my_entry['layers'][1]}, "
              f"model: {model_name}")

        # ── Check 1: same pipeline already running? Route as warm query ──
        with self.peer_lock:
            existing_peer = self.peers.get(model_name)

        if existing_peer is not None and existing_peer.shared.pipeline == shared.pipeline:
            print(f"[Daemon] Same pipeline already active for {model_name} — routing as warm query")
            self._run_query_on_existing_peer(conn, query, model_name)
            return

        # ── Check 2: another thread already creating this model? Wait for it ──
        with self.peer_lock:
            if model_name in self._creating:
                creating = True
            else:
                self._creating.add(model_name)
                creating = False

        if creating:
            print(f"[Daemon] Another thread creating {model_name} — waiting...")
            while True:
                with self.peer_lock:
                    peer = self.peers.get(model_name)
                if peer is not None and peer.model is not None:
                    break
                time.sleep(0.5)
            print(f"[Daemon] {model_name} ready — routing as warm query")
            self._run_query_on_existing_peer(conn, query, model_name)
            return

        # ── Full rebuild: tear down old, create new ──
        try:
            with self.peer_lock:
                if model_name in self.schedulers:
                    print(f"[Daemon] Draining scheduler for {model_name} before rebuild...")
                    self.schedulers[model_name].drain()
                    self.schedulers[model_name].stop()
                    del self.schedulers[model_name]
                    print(f"[Daemon] Scheduler torn down")
                if model_name in self.peers:
                    try:
                        self.peers[model_name].send_stop()
                    except Exception:
                        pass
                    self.peers[model_name].cleanup()
                    del self.peers[model_name]
                    del self._peer_last_used[model_name]
                if model_name in self._gen_threads:
                    del self._gen_threads[model_name]

                self._ensure_memory_headroom()

            # create peer, load model
            peer = InferencePeer(shared, self.local)
            peer.load_query_into_model(query)

            with self.peer_lock:
                self.peers[model_name] = peer
                self._peer_last_used[model_name] = time.time()

        finally:
            with self.peer_lock:
                self._creating.discard(model_name)

        # Phase 2: report ready, wait for start signal
        send_message(conn, MSG_READY)
        print(f"[Daemon] Model loaded — waiting for start signal")

        msg_type, _ = read_message(conn)
        if msg_type != MSG_START:
            print(f"[Daemon] Expected MSG_START, got {msg_type}")
            return

        # Phase 3: connect to chain neighbors
        is_master = my_entry["role"] == "master"
        print(f"[Daemon] Start signal received — connecting chain")
        peer.connect()

        if is_master:
            # Create Scheduler and start it in a dedicated thread
            scheduler = Scheduler(peer)
            sched_thread = threading.Thread(target=scheduler.run, daemon=True)
            sched_thread.start()

            with self.peer_lock:
                self.schedulers[model_name] = scheduler

            print(f"[Daemon] Scheduler started for {model_name}")

            # Submit the first query through the Scheduler
            session = Session(session_id=query.session_id)
            if query.messages:
                session.messages = query.messages
            else:
                session.add_user_message(query.prompt)

            cache_key = (query.session_id, model_name)
            request = InFlightRequest(query, session, cache_key)

            print(f"[Scheduler] Submitting first request (session={query.session_id})")
            scheduler.submit(request)
            self._stream_and_respond(conn, request, label="First query")
        else:
            # Worker/tail: start long-running generation loop in a thread
            gen_thread = threading.Thread(
                target=peer.run_generation,
                kwargs={"query": query},
                daemon=True,
            )
            gen_thread.start()
            self._gen_threads[model_name] = gen_thread
            print(f"[Daemon] {my_entry['role']} generation loop started (long-running)")

    def _handle_query(self, conn, payload):
        """
        Handle a warm query (MSG_QUERY). Looks up existing peer,
        routes through _run_query_on_existing_peer.
        """
        from user_query import UserQuery

        query = from_bytes(UserQuery, payload)
        model_name = query.model_name

        with self.peer_lock:
            peer = self.peers.get(model_name)
            if peer is not None:
                self._peer_last_used[model_name] = time.time()

        if peer is None or peer.model is None:
            print(f"[Daemon] No peer for model '{model_name}' — rejecting MSG_QUERY")
            send_message(conn, MSG_QUERY_FAIL)
            return

        self._run_query_on_existing_peer(conn, query, model_name)

    def _run_query_on_existing_peer(self, conn, query, model_name):
        """
        Shared logic for running a query on an already-loaded peer.
        Called from both _handle_query (MSG_QUERY) and _handle_config_query
        (MSG_CONFIG with matching pipeline — routed as warm).
        """
        from session import Session
        from scheduler import InFlightRequest

        with self.peer_lock:
            peer = self.peers.get(model_name)
            scheduler = self.schedulers.get(model_name)

        if peer is None or peer.model is None:
            print(f"[Daemon] No peer for {model_name} — rejecting")
            send_message(conn, MSG_QUERY_FAIL)
            return

        is_master = peer.is_master

        # Synchronize: report ready, wait for start
        send_message(conn, MSG_READY)
        print(f"[Daemon] Warm query ready (model={model_name}, role={peer.role})")

        msg_type, _ = read_message(conn)
        if msg_type != MSG_START:
            print(f"[Daemon] Expected MSG_START, got {msg_type}")
            return

        if is_master:
            # The peer may exist while its creating thread is still in
            # connect() or hasn't created the Scheduler yet (concurrent
            # cold starts). Wait for the scheduler to come up rather
            # than rejecting — the creating thread is actively building it.
            waited = 0.0
            while scheduler is None and waited < 120.0:
                if waited == 0.0:
                    print(f"[Daemon] Scheduler for {model_name} not ready yet — "
                          f"waiting for pipeline setup to complete...")
                time.sleep(0.5)
                waited += 0.5
                with self.peer_lock:
                    scheduler = self.schedulers.get(model_name)

            if scheduler is None:
                print(f"[Daemon] Scheduler for {model_name} never came up — rejecting")
                send_message(conn, MSG_QUERY_FAIL)
                return

            if waited > 0:
                print(f"[Daemon] Scheduler ready after {waited:.1f}s — proceeding")

            # Build session from query messages
            session = Session(session_id=query.session_id)
            if query.messages:
                session.messages = query.messages
            else:
                session.add_user_message(query.prompt)

            cache_key = (query.session_id, model_name)
            request = InFlightRequest(query, session, cache_key)

            # Retry submit in case scheduler is draining
            while True:
                try:
                    print(f"[Scheduler] Submitting request "
                          f"(session={query.session_id}, active={scheduler.active_count()} in-flight)")
                    scheduler.submit(request)
                    break
                except RuntimeError:
                    print(f"[Scheduler] Draining — retrying in 0.5s...")
                    time.sleep(0.5)
                    with self.peer_lock:
                        scheduler = self.schedulers.get(model_name)
                    if scheduler is None:
                        print(f"[Daemon] Scheduler gone after drain — rejecting")
                        send_message(conn, MSG_QUERY_FAIL)
                        return

            self._stream_and_respond(conn, request, label="Query")
        else:
            # Worker/tail already in their long-running loops — nothing to do
            print(f"[Daemon] Warm query ack — {peer.role} loop already running")

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

    def _stream_and_respond(self, conn, request, label="Query"):
        """
        Forward text deltas to the orchestrator as they are generated,
        then send the final MSG_RESPONSE. If the orchestrator hangs up
        mid-stream, generation still completes (the cache stays valid).
        """
        import queue as _queue

        streamed = 0
        while True:
            try:
                delta = request.token_queue.get(timeout=0.5)
            except _queue.Empty:
                if request.done_event.is_set():
                    break
                continue

            if delta is None:      # sentinel from Scheduler
                break

            try:
                send_message(conn, MSG_TOKEN_STREAM, delta.encode("utf-8"))
                streamed += 1
            except Exception as e:
                print(f"[Daemon] Stream broken ({e}) — finishing generation anyway")
                break

        request.done_event.wait()
        response = request.result or ""
        print(f"[Daemon] {label} complete — streamed {streamed} deltas, "
              f"sending MSG_RESPONSE ({len(response)} chars)")
        try:
            send_message(conn, MSG_RESPONSE, response.encode("utf-8"))
            print(f"[Daemon] MSG_RESPONSE sent")
        except Exception as e:
            print(f"[Daemon] Could not send final response: {e}")

    def _evict_lru_peer(self):
        """Evict the least recently used model to free memory.
        Drains the Scheduler first so in-flight requests complete
        before the peer is torn down."""
        if not self._peer_last_used:
            return
        oldest = min(self._peer_last_used, key=self._peer_last_used.get)
        print(f"[Daemon] Evicting LRU model '{oldest}'")

        # Drain and stop Scheduler if one exists (master node)
        if oldest in self.schedulers:
            print(f"[Daemon] Draining scheduler for '{oldest}' before eviction...")
            self.schedulers[oldest].drain()
            self.schedulers[oldest].stop()
            del self.schedulers[oldest]
            print(f"[Daemon] Scheduler drained and stopped")

        # Send MSG_STOP to break worker/tail loops
        try:
            self.peers[oldest].send_stop()
        except Exception:
            pass

        self.peers[oldest].cleanup()
        del self.peers[oldest]
        del self._peer_last_used[oldest]

        if oldest in self._gen_threads:
            del self._gen_threads[oldest]


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
