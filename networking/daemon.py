import os, sys, json, socket, threading, time, torch, io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MSG_LOG_REQ, MSG_LOG_RESP,
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
        finally:
            self.shutdown()
            server.close()

    def _run_generation_guarded(self, peer, query, model_name, role):
        """
        Run a worker/tail generation loop and release the peer when it ends.

        Without this wrapper, an upstream node exiting raises ConnectionError
        deep inside receive_hidden, the thread dies with a traceback, and the
        peer is left loaded with dead sockets. Later queries would then match
        the cached pipeline and route to a loop that no longer exists.
        """
        try:
            peer.run_generation(query=query)
            print(f"[Daemon] {role} loop for '{model_name}' ended normally")
        except ConnectionError as e:
            print(f"[Daemon] Lost connection to the pipeline ({e})")
        except Exception as e:
            print(f"[Daemon] {role} loop for '{model_name}' failed: "
                  f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._release_peer(model_name)

    def _release_peer(self, model_name):
        """
        Drop a peer whose pipeline is no longer usable, so the next query
        rebuilds instead of reusing dead sockets.
        """
        with self.peer_lock:
            peer = self.peers.pop(model_name, None)
            scheduler = self.schedulers.pop(model_name, None)
            self._peer_last_used.pop(model_name, None)
            self._gen_threads.pop(model_name, None)

        if peer is None and scheduler is None:
            return

        # Registries are already updated, so this runs unlocked: in-flight
        # requests finish while other models keep serving.
        if scheduler is not None:
            scheduler.drain(timeout=30.0)
            scheduler.stop()
        if peer is not None:
            try:
                peer.send_stop()
            except Exception:
                pass
            try:
                peer.cleanup()
            except Exception:
                pass
        if self.local.device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"[Daemon] Released '{model_name}' — the next query will rebuild it")

    def _peer_is_healthy(self, model_name):
        """
        Whether a loaded peer can still serve a query. A peer with a loaded
        model but a dead generation loop or dead scheduler is worse than no
        peer at all: it accepts the query and then never responds.
        """
        with self.peer_lock:
            peer = self.peers.get(model_name)
            scheduler = self.schedulers.get(model_name)
            gen_thread = self._gen_threads.get(model_name)

        if peer is None or peer.model is None:
            return False

        if peer.is_master:
            if scheduler is None or not scheduler.running:
                print(f"[Daemon] '{model_name}' has no live scheduler")
                return False
        else:
            if gen_thread is None or not gen_thread.is_alive():
                print(f"[Daemon] '{model_name}' generation loop is no longer running")
                return False

        return True

    def shutdown(self):
        """
        Stop accepting work and tell connected peers before the process
        goes away, so they release their peers cleanly rather than
        discovering the loss as an abrupt socket close.
        """
        if not self.running and not self.peers:
            return
        print("[Daemon] Shutting down — notifying peers")
        self.running = False

        for model_name in list(self.peers.keys()):
            peer = self.peers.get(model_name)
            if peer is None:
                continue
            try:
                peer.send_stop()
                print(f"[Daemon] Sent stop for '{model_name}'")
            except Exception:
                pass

        for model_name in list(self.peers.keys()):
            self._release_peer(model_name)

        print("[Daemon] Shutdown complete")

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
            elif msg_type == MSG_LOG_REQ:
                self._handle_log_request(conn, payload)
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

    def _handle_log_request(self, conn, payload):
        """
        Return console output this machine has produced since a given
        sequence number. Each node's logs live only on that node, so the
        UI asks every node in the pipeline directly.
        """
        from logbuffer import BUFFER
        try:
            since = int(payload.decode("utf-8") or 0)
        except (ValueError, UnicodeDecodeError):
            since = 0

        response = json.dumps({
            "ip": self.local.tailscale_ip,
            "device": self.local.device,
            "lines": BUFFER.read(since=since),
            "latest_seq": BUFFER.latest_seq(),
        })
        send_message(conn, MSG_LOG_RESP, response.encode("utf-8"))

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
            if self._peer_is_healthy(model_name):
                print(f"[Daemon] Same pipeline already active for {model_name} "
                      f"— routing as warm query")
                self._run_query_on_existing_peer(conn, query, model_name)
                return
            print(f"[Daemon] Pipeline for {model_name} matches but is no longer "
                  f"usable — rebuilding")
            self._release_peer(model_name)

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
        #
        # None of this holds peer_lock. Draining and model loading take
        # seconds to minutes, and holding the lock across them would stall
        # every query for every *other* model behind this one.
        try:
            self._release_peer(model_name)
            self._make_room_for(model_name)

            print(f"[Daemon] Loading '{model_name}'...")
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
            # Worker/tail: start long-running generation loop in a thread.
            # Wrapped so an upstream node going away releases this peer
            # instead of silently killing the thread.
            gen_thread = threading.Thread(
                target=self._run_generation_guarded,
                args=(peer, query, model_name, my_entry["role"]),
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

        if not self._peer_is_healthy(model_name):
            print(f"[Daemon] Peer for '{model_name}' is stale — rejecting so the "
                  f"orchestrator falls back to a cold rebuild")
            self._release_peer(model_name)
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

    def _peer_is_busy(self, model_name, peer):
        """
        Whether this model is mid-generation. Evicting a busy peer would
        cut off a conversation in progress and force the next query to
        cold start, so busy peers are left alone.
        """
        scheduler = self.schedulers.get(model_name)
        if scheduler is not None:
            return scheduler.active_count() > 0
        # Worker/tail have no scheduler; recent hidden states mean tokens
        # are still flowing through this node.
        return (time.time() - getattr(peer, "last_activity", 0.0)) < 5.0

    def _take_eviction_candidate(self, exclude=None):
        """
        Under the lock, pick the least recently used *idle* model and
        remove it from the registries so nobody else can route to it.

        Returns (model_name, peer, scheduler) or None if every remaining
        model is busy. The caller tears the peer down outside the lock —
        draining can take seconds and must not block other models.
        """
        with self.peer_lock:
            idle = [
                name for name, peer in self.peers.items()
                if name != exclude and not self._peer_is_busy(name, peer)
            ]
            if not idle:
                return None

            oldest = min(idle, key=lambda n: self._peer_last_used.get(n, 0.0))
            peer = self.peers.pop(oldest, None)
            scheduler = self.schedulers.pop(oldest, None)
            self._peer_last_used.pop(oldest, None)
            self._gen_threads.pop(oldest, None)
            return oldest, peer, scheduler

    def _teardown_peer(self, model_name, peer, scheduler):
        """Drain and release an already-unregistered peer. Never holds the lock."""
        if scheduler is not None:
            scheduler.drain(timeout=30.0)
            scheduler.stop()
        if peer is not None:
            try:
                peer.send_stop()
            except Exception:
                pass
            try:
                peer.cleanup()
            except Exception:
                pass
        if self.local.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[Daemon] Unloaded '{model_name}'")

    def _make_room_for(self, model_name, timeout=90.0):
        """
        Free memory for a model about to load, without holding peer_lock
        while waiting. Idle models are unloaded first; a model that is
        actively generating is waited for rather than interrupted.
        """
        deadline = time.time() + timeout
        announced = False

        while True:
            total, free = self._get_memory_status()
            required = int(total * self.local.overhead)
            if free >= required:
                return

            if not announced:
                print(f"[Daemon] Need {required/1e9:.1f}GB free to load "
                      f"'{model_name}', have {free/1e9:.1f}GB")
                announced = True

            victim = self._take_eviction_candidate(exclude=model_name)
            if victim is not None:
                name, peer, scheduler = victim
                print(f"[Daemon] Unloading idle model '{name}' to make room")
                self._teardown_peer(name, peer, scheduler)
                continue

            # Everything still loaded is mid-generation. Wait for it to
            # finish rather than cutting off a conversation.
            if time.time() >= deadline:
                print(f"[Daemon] Still short on memory after {timeout:.0f}s "
                      f"— loading '{model_name}' anyway")
                return
            time.sleep(0.5)

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
