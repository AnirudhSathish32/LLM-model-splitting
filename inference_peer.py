import torch
import time
import socket
import struct

from model import Model
from networking.serialization import tensor_to_bytes, tensor_from_bytes, to_bytes, from_bytes
from networking.protocol import send_message, read_message
from config import (
    MSG_FIRST_PASS, MSG_NEXT_PASS, MSG_STOP, MSG_LAYER,
    MSG_EOS, MSG_TTFT, MSG_TOKEN, MSG_RESPONSE,
    SharedConfig, LocalConfig,
)


class InferencePeer:
    def __init__(self, shared: SharedConfig, local: LocalConfig):
        self.shared = shared
        self.local = local

        # find our entry in the pipeline
        my_assignment = None
        my_index = None
        my_ip = local.tailscale_ip

        for i, entry in enumerate(shared.pipeline):
            if entry["ip"] == my_ip:
                my_assignment = entry
                my_index = i
                break

        if my_assignment is None:
            raise ValueError(f"This machine ({my_ip}) is not in the pipeline")

        # identity
        self.my_pipeline_entry = my_assignment
        self.role = my_assignment["role"]
        self.is_master = self.role == "master"
        self.is_tail = self.role == "tail"

        # ── circular chain neighbors ──
        # Master → Worker(s) → Tail → Master
        # Master's upstream is Tail (receives tokens back).
        # Tail's downstream is Master (sends tokens back).
        n = len(shared.pipeline)

        if my_index > 0:
            self.upstream_ip = shared.pipeline[my_index - 1]["ip"]
        else:
            # master: upstream wraps to tail
            self.upstream_ip = shared.pipeline[n - 1]["ip"]

        if my_index < n - 1:
            self.downstream_ip = shared.pipeline[my_index + 1]["ip"]
        else:
            # tail: downstream wraps to master
            self.downstream_ip = shared.pipeline[0]["ip"]

        self.initiator_ip = shared.initiator_ip

        # model
        self.model = None
        self.loaded_model_name = None

        # caches keyed by (session_id, model_name)
        self.caches = {}
        self._active_cache_key = None

        # connections (set by connect() or injected for testing)
        self.upstream_conn = None
        self.downstream_conn = None

    # ================================================================
    # CACHE MANAGEMENT
    # ================================================================

    def _get_cache(self):
        """Get the cache for the active session+model, or None for first use."""
        if self._active_cache_key is None:
            return None
        return self.caches.get(self._active_cache_key)

    def _set_cache(self, cache):
        """Store the cache for the active session+model."""
        if self._active_cache_key is not None:
            self.caches[self._active_cache_key] = cache

    # ================================================================
    # MODEL LOADING
    # ================================================================

    def load_query_into_model(self, query):
        """
        Load the model slice if needed, set up the cache key for this query.
        """
        if self.loaded_model_name != query.model_name:
            # different model — unload old, load new
            if self.model is not None:
                self.model.unload()

            self.model = Model(
                model_name=query.model_name,
                role=self.role,
                layer_start=self.my_pipeline_entry["layers"][0],
                layer_end=self.my_pipeline_entry["layers"][1],
                local_config=self.local,
                dtype=query.dtype,
            )
            self.model.load()
            self.model.register_hooks(debug=self.shared.debug)
            self.loaded_model_name = query.model_name
            self.caches.clear()

        # set up cache key for this session
        cache_key = (query.session_id, query.model_name)
        if cache_key not in self.caches:
            self.caches[cache_key] = None
        self._active_cache_key = cache_key

        self.model.reset_turn_state()

    # ================================================================
    # CONNECTION — circular chain with retry
    # ================================================================

    def connect(self, max_retries=10, retry_delay=0.5):
        """
        Establish chain connections. Circular topology:
          Master → Worker(s) → Tail → Master
 
        For 2-machine (master + tail): single full-duplex TCP connection.
        Both hidden states (master→tail) and tokens (tail→master) flow
        on the same socket. No second port needed.
 
        For N-machine (with workers): each adjacent pair gets one connection.
        Tail also connects back to master for the token return channel.
        """
        port = self.shared.port  # 65432
        n_peers = len(self.shared.pipeline)
 
        if self.is_master:
            # Accept connection from first downstream neighbor
            self.downstream_conn = self._listen_accept(port)
            print(f"[Peer] Master: downstream connected on port {port}")
 
            if n_peers == 2:
                # 2-machine: reuse same connection for token return (full-duplex)
                self.upstream_conn = self.downstream_conn
                print(f"[Peer] Master: upstream = downstream (full-duplex, 2-machine)")
            else:
                # N-machine: tail connects separately for token return
                self.upstream_conn = self._listen_accept(port)
                print(f"[Peer] Master: upstream (token return) accepted on port {port}")
 
        elif self.is_tail:
            # Connect upstream to previous node in chain
            self.upstream_conn = self._connect_with_retry(
                self.upstream_ip, port, max_retries, retry_delay)
            print(f"[Peer] Tail: connected upstream to {self.upstream_ip}:{port}")
 
            if n_peers == 2:
                # 2-machine: reuse same connection for token return (full-duplex)
                self.downstream_conn = self.upstream_conn
                print(f"[Peer] Tail: downstream = upstream (full-duplex, 2-machine)")
            else:
                # N-machine: separate connection back to master for tokens
                self.downstream_conn = self._connect_with_retry(
                    self.downstream_ip, port, max_retries, retry_delay)
                print(f"[Peer] Tail: connected downstream (token return) to {self.downstream_ip}:{port}")
 
        else:
            # Worker: connect upstream, listen for downstream
            self.upstream_conn = self._connect_with_retry(
                self.upstream_ip, port, max_retries, retry_delay)
            print(f"[Peer] Worker: connected upstream to {self.upstream_ip}:{port}")
 
            self.downstream_conn = self._listen_accept(port)
            print(f"[Peer] Worker: downstream connected on port {port}")

    def _listen_accept(self, port):
        """Bind, listen, accept one connection. SO_REUSEADDR + TCP_NODELAY."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(1)
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.close()
        print(f"[Peer] Accepted connection from {addr} on port {port}")
        return conn

    def _connect_with_retry(self, ip, port, max_retries, retry_delay):
        """Connect to a peer with exponential backoff retry. TCP_NODELAY set."""
        delay = retry_delay
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((ip, port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"[Peer] Connection to {ip}:{port} refused, "
                          f"retry {attempt+1}/{max_retries} in {delay:.1f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 10.0)  # cap at 10s
                else:
                    raise ConnectionError(
                        f"Failed to connect to {ip}:{port} after {max_retries} attempts")

    # ================================================================
    # WIRE METHODS
    # ================================================================

    def send_hidden(self, hidden, msg_type=MSG_FIRST_PASS):
        """
        Send hidden state to downstream neighbor.
        Prepends session_id so the receiver can switch to the correct KV cache.
        Format: [session_id_len:2][session_id:N][tensor_bytes]
        """
        session_id = self._active_cache_key[0] if self._active_cache_key else ""
        sid_bytes = session_id.encode("utf-8")
        tensor_bytes = tensor_to_bytes(hidden)
        payload = struct.pack(">H", len(sid_bytes)) + sid_bytes + tensor_bytes
        send_message(self.downstream_conn, msg_type, payload)

    def receive_hidden(self):
        """
        Receive hidden state from upstream neighbor.
        Extracts session_id, switches active cache, and casts to the
        local model's compute dtype (handles cross-dtype pipelines).
        Returns (msg_type, hidden_tensor).
        """
        msg_type, payload = read_message(self.upstream_conn)
        if msg_type == MSG_STOP:
            return MSG_STOP, None

        # Extract session_id prefix
        sid_len = struct.unpack(">H", payload[:2])[0]
        session_id = payload[2:2 + sid_len].decode("utf-8")
        tensor_bytes = payload[2 + sid_len:]

        # Switch active cache to this session
        cache_key = (session_id, self.loaded_model_name)
        if cache_key not in self.caches:
            self.caches[cache_key] = None
        self._active_cache_key = cache_key

        # Deserialize and cast to local model's dtype
        hidden = tensor_from_bytes(tensor_bytes, device=self.model.device)
        hidden = hidden.to(dtype=self.model.dtype)
        return msg_type, hidden

    def send_token(self, token):
        """Send a generated token ID downstream (tail → master)."""
        payload = token.cpu().numpy().tobytes()
        send_message(self.downstream_conn, MSG_TOKEN, payload)

    def receive_token(self):
        """
        Receive token or EOS from upstream (tail → master path).
        Returns ("token", token_tensor) or ("eos", None).
        """
        msg_type, payload = read_message(self.upstream_conn)
        if msg_type == MSG_EOS:
            return "eos", None
        if msg_type == MSG_TOKEN:
            token = torch.frombuffer(bytearray(payload), dtype=torch.int64)
            return "token", token
        raise ValueError(f"Expected MSG_TOKEN or MSG_EOS, got {msg_type}")

    def send_eos(self):
        """Send end-of-sequence downstream (tail → master)."""
        send_message(self.downstream_conn, MSG_EOS)

    def send_stop(self):
        """Send stop signal downstream (master → workers on EOS)."""
        send_message(self.downstream_conn, MSG_STOP)

    def send_response(self, response_text):
        """Send the final response string to the initiator."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.initiator_ip, self.shared.port))
        send_message(sock, MSG_RESPONSE, response_text.encode("utf-8"))
        sock.close()

    # ================================================================
    # GENERATION LOOPS
    # ================================================================

    def run_generation(self, query=None, session=None):
        """Dispatch to role-specific generation loop."""
        self._drain_stale()
        if self.is_master:
            return self.run_master_generation(query, session)
        elif self.is_tail:
            return self.run_tail_generation(query)
        else:
            return self.run_worker_generation()
        
    def _drain_stale(self):
        """
        Discard leftover messages in socket buffers from previous generation.
 
        After a generation ends, MSG_STOP or MSG_EOS may sit unread in the
        buffers (e.g. tail sends EOS, master sends STOP, but neither reads
        the other's final message). Without draining, the next generation
        would read stale data and immediately break.
        """
        seen = set()
        for conn in (self.upstream_conn, self.downstream_conn):
            if conn is None or id(conn) in seen:
                continue
            seen.add(id(conn))
            conn.setblocking(False)
            try:
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    print(f"[Peer] Drained {len(data)} stale bytes")
            except (BlockingIOError, OSError):
                pass
            finally:
                conn.setblocking(True)

    def run_master_generation(self, query, session=None):
        """
        Blocking master generation — runs all steps to completion.
        Used by daemon for single-query mode (no Scheduler).
        Sends MSG_STOP when done to tear down the chain.
        """
        from scheduler import InFlightRequest

        request = InFlightRequest(query, session, self._active_cache_key)
        self.prepare_request(request)

        ttft_start = time.perf_counter()
        ttft = None

        while self.step_master(request):
            if ttft is None:
                ttft = time.perf_counter() - ttft_start

        if ttft is None:
            ttft = time.perf_counter() - ttft_start

        # Single-query mode: stop the chain now
        self.send_stop()

        response = self.model.decode(request.generated_ids)

        print(f"[Master] Generation complete — {request.token_count} tokens")
        print(f"[Master] TTFT: {ttft*1000:.1f}ms, "
              f"total: {(time.perf_counter()-ttft_start)*1000:.1f}ms")

        return response

    def prepare_request(self, request):
        """
        Tokenize and set up first pass for a new request.
        Called once per request, before the first step_master call.
        """
        self._drain_stale()
        self._active_cache_key = request.cache_key

        # Get existing cache for this session+model (None on first query)
        request.cache = self._get_cache()

        # Tokenize full conversation
        if request.session is not None:
            messages = request.session.messages
        else:
            messages = [{"role": "user", "content": request.query.prompt}]

        request.full_sequence_ids = self.model.tokenize(messages).to(self.model.device)

        # Warm cache: skip tokens already cached
        cache_len = request.cache.get_seq_length() if request.cache is not None else 0
        total_len = request.full_sequence_ids.shape[1]

        if cache_len > 0 and cache_len < total_len:
            request.first_pass_input = request.full_sequence_ids[:, cache_len:]
            print(f"[Master] Warm cache: {cache_len} cached, "
                  f"{total_len - cache_len} new tokens to prefill")
        elif cache_len >= total_len and cache_len > 0:
            print(f"[Master] Cache invalidated: cache_len={cache_len} >= input_len={total_len}")
            request.cache = None
            request.first_pass_input = request.full_sequence_ids
        else:
            request.first_pass_input = request.full_sequence_ids

        request.first_pass = True
        request.token_count = 0
        request.generated_ids = []

        print(f"[Master] Request prepared — {total_len} input tokens, "
              f"max {request.query.tokens_to_generate} to generate")

    def step_master(self, request):
        """
        One forward step for the master. Advances the request by one token.

        Returns True if the request needs more steps, False if done
        (EOS received or max tokens reached).

        Does NOT send MSG_STOP — the caller (run_master_generation or
        Scheduler) decides when to stop the pipeline.
        """
        import torch

        sid = request.query.session_id
        tc = request.token_count

        # Activate this request's cache
        self._active_cache_key = request.cache_key
        self.model.pass_counter["i"] = tc

        # Determine input
        if request.first_pass:
            model_input = request.first_pass_input
        else:
            model_input = request.full_sequence_ids[:, -1:]

        # Forward through master layers
        print(f"[Token {tc}] ({sid}) forward (input shape {model_input.shape})...", flush=True)
        hidden, request.cache = self.model.forward(model_input, request.cache)
        print(f"[Token {tc}] ({sid}) forward done → sending hidden downstream", flush=True)

        # Send hidden downstream
        msg_type = MSG_FIRST_PASS if request.first_pass else MSG_NEXT_PASS
        self.send_hidden(hidden, msg_type)
        request.first_pass = False
        print(f"[Token {tc}] ({sid}) waiting for token from tail...", flush=True)

        # Receive token from tail
        msg_string, token = self.receive_token()
        print(f"[Token {tc}] ({sid}) received '{msg_string}'", flush=True)

        if msg_string == "eos":
            print(f"[Token {tc}] ({sid}) EOS — request complete", flush=True)
            self.caches[request.cache_key] = request.cache
            return False

        # Append token and update state
        token_id = token.item()
        request.generated_ids.append(token_id)
        decoded = self.model.decode(request.generated_ids)
        print(f"[Token {tc}] ({sid}) id={token_id}, text so far: '{decoded}'", flush=True)

        # Push incremental delta for streaming consumers (UI/API)
        delta = decoded[len(request._last_decoded):]
        request._last_decoded = decoded
        if delta:
            request.token_queue.put(delta)

        request.full_sequence_ids = torch.cat(
            [request.full_sequence_ids, token.unsqueeze(0).to(request.full_sequence_ids.device)],
            dim=-1,
        )
        request.token_count += 1

        if request.token_count >= request.query.tokens_to_generate:
            print(f"[Token {tc}] ({sid}) max tokens reached — request complete", flush=True)
            self.caches[request.cache_key] = request.cache
            return False

        return True

    def run_worker_generation(self):
        """
        Worker loop with per-step cache switching.
        receive_hidden extracts the session_id and sets _active_cache_key,
        so each step uses the correct cache for the current request.
        """
        while True:
            msg_type, hidden = self.receive_hidden()

            if msg_type == MSG_STOP:
                self.send_stop()
                break

            # _active_cache_key was set by receive_hidden
            cache = self._get_cache()

            hidden = hidden.to(self.model.device)
            self.model.pass_counter["i"] += 1

            hidden, cache = self.model.forward(hidden, cache)

            self._set_cache(cache)
            self.send_hidden(hidden, msg_type)

    def run_tail_generation(self, query=None):
        """
        Long-running tail loop. Processes hidden states continuously,
        switching caches via session tags from receive_hidden.

        On EOS: sends MSG_EOS to master and continues (does NOT break).
        On MSG_STOP: breaks and exits (pipeline shutdown).

        The master controls max_tokens — the tail just processes
        whatever hidden states arrive.
        """
        token_count = 0

        print(f"[Tail] Starting generation loop (long-running)")

        while True:
            msg_type, hidden = self.receive_hidden()

            if msg_type == MSG_STOP:
                print(f"[Tail] Received MSG_STOP — shutting down")
                break

            # _active_cache_key was set by receive_hidden
            cache = self._get_cache()

            hidden = hidden.to(self.model.device)
            self.model.pass_counter["i"] = token_count

            token, cache = self.model.forward(hidden, cache)

            self._set_cache(cache)

            # check EOS
            eos_ids = self.model.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]

            if token.item() in eos_ids:
                print(f"[Tail] EOS token detected — sending EOS (staying in loop)")
                self.send_eos()
                continue    # ← stay in loop, master decides what's next

            self.send_token(token)
            token_count += 1

        print(f"[Tail] Loop exited — {token_count} total tokens generated")

    # ================================================================
    # CLEANUP
    # ================================================================

    def cleanup(self):
        """Close connections and unload model."""
        for conn in (self.upstream_conn, self.downstream_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        self.upstream_conn = None
        self.downstream_conn = None

        if self.model is not None:
            self.model.unload()
            self.model = None
