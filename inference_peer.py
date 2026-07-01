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
        """Bind, listen, accept one connection. SO_REUSEADDR applied."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(1)
        conn, addr = server.accept()
        server.close()
        print(f"[Peer] Accepted connection from {addr} on port {port}")
        return conn

    def _connect_with_retry(self, ip, port, max_retries, retry_delay):
        """Connect to a peer with exponential backoff retry."""
        delay = retry_delay
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((ip, port))
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
        """Send hidden state to downstream neighbor."""
        payload = tensor_to_bytes(hidden)
        send_message(self.downstream_conn, msg_type, payload)

    def receive_hidden(self):
        """
        Receive hidden state from upstream neighbor.
        Returns (msg_type, hidden_tensor).
        msg_type is MSG_FIRST_PASS, MSG_NEXT_PASS, or MSG_STOP.
        """
        msg_type, payload = read_message(self.upstream_conn)
        if msg_type == MSG_STOP:
            return MSG_STOP, None
        hidden = tensor_from_bytes(payload, device=self.model.device)
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
        if self.is_master:
            return self.run_master_generation(query, session)
        elif self.is_tail:
            return self.run_tail_generation(query)
        else:
            return self.run_worker_generation()

    def run_master_generation(self, query, session=None):
        """
        Master loop:
        1. Tokenize full conversation → input_ids
        2. If cache is warm, skip tokens already in cache (feed only new ones)
        3. Forward through master layers (StopIteration captures hidden)
        4. Send hidden downstream
        5. Receive token from upstream (tail → master circular path)
        6. Append token, repeat until EOS or max tokens
        7. On EOS: send MSG_STOP downstream to unblock workers
        """
        cache = self._get_cache()

        # Build input from session messages if available, else raw prompt
        if session is not None:
            messages = session.messages
        else:
            messages = [{"role": "user", "content": query.prompt}]

        print("[Master] Attempting to Tokenize")
        full_sequence_ids = self.model.tokenize(messages).to(self.model.device)
        print("[Master] Tokenize succeessfuly")

        # ── Warm cache: skip tokens the cache already covers ──
        cache_len = cache.get_seq_length() if cache is not None else 0
        total_len = full_sequence_ids.shape[1]

        if cache_len > 0 and cache_len < total_len:
            # Resume — only prefill the new tokens
            first_pass_input = full_sequence_ids[:, cache_len:]
            print(f"[Master] Warm cache: {cache_len} cached, "
                  f"{total_len - cache_len} new tokens to prefill")
        elif cache_len >= total_len and cache_len > 0:
            # Cache covers everything or more — shouldn't happen, invalidate
            print(f"[Master] Cache invalidated: cache_len={cache_len} >= input_len={total_len}")
            cache = None
            first_pass_input = full_sequence_ids
        else:
            # Cold start
            first_pass_input = full_sequence_ids

        first_pass = True
        token_count = 0
        ttft_start = time.perf_counter()
        ttft = None

        print(f"[Master] Starting generation loop — max {query.tokens_to_generate} tokens, "
              f"input shape: {full_sequence_ids.shape}")

        while token_count < query.tokens_to_generate:
            self.model.pass_counter["i"] = token_count

            if first_pass:
                model_input = first_pass_input
            else:
                model_input = full_sequence_ids[:, -1:]

            print(f"[Master] Token {token_count}: forward (input shape {model_input.shape})...")
            hidden, cache = self.model.forward(model_input, cache)
            print(f"[Master] Token {token_count}: forward done, hidden shape {hidden.shape}")

            msg_type = MSG_FIRST_PASS if first_pass else MSG_NEXT_PASS
            print(f"[Master] Token {token_count}: sending hidden to tail...")
            self.send_hidden(hidden, msg_type)
            first_pass = False
            print(f"[Master] Token {token_count}: hidden sent, waiting for token from tail...")

            # receive token from tail (via circular upstream connection)
            msg_string, token = self.receive_token()
            print(f"[Master] Token {token_count}: received '{msg_string}' from tail")

            if ttft is None:
                ttft = time.perf_counter() - ttft_start

            if msg_string == "eos":
                # propagate stop down the chain so workers unblock
                print(f"[Master] EOS received — sending stop downstream")
                self.send_stop()
                break

            self.model.generated_ids.append(token.item())
            decoded_so_far = self.model.decode(self.model.generated_ids)
            print(f"[Master] Token {token_count}: id={token.item()}, "
                  f"text so far: '{decoded_so_far}'")
            full_sequence_ids = torch.cat(
                [full_sequence_ids, token.unsqueeze(0).to(full_sequence_ids.device)],
                dim=-1,
            )
            token_count += 1

        # If we hit max tokens without EOS, still stop the pipeline
        if token_count >= query.tokens_to_generate:
            print(f"[Master] Max tokens reached — sending stop downstream")
            self.send_stop()

        self._set_cache(cache)
        response = self.model.decode(self.model.generated_ids)
        print(f"[Master] Generation complete — {token_count} tokens")

        if ttft is not None:
            print(f"[Master] TTFT: {ttft*1000:.1f}ms, "
                  f"tokens: {token_count}, total: {(time.perf_counter()-ttft_start)*1000:.1f}ms")

        return response

    def run_worker_generation(self):
        """
        Worker loop:
        1. Receive hidden from upstream
        2. Forward through worker layers
        3. Send hidden downstream
        4. Break on MSG_STOP
        """
        cache = self._get_cache()

        while True:
            msg_type, hidden = self.receive_hidden()

            if msg_type == MSG_STOP:
                # propagate stop to next node in chain
                self.send_stop()
                break

            hidden = hidden.to(self.model.device)
            self.model.pass_counter["i"] += 1

            hidden, cache = self.model.forward(hidden, cache)

            self.send_hidden(hidden, msg_type)

        self._set_cache(cache)

    def run_tail_generation(self, query=None):
        """
        Tail loop:
        1. Receive hidden from upstream
        2. Forward through tail layers + lm_head → token
        3. Send token downstream (back to master via circular path)
        4. Break on EOS or max tokens, or MSG_STOP from upstream
        """
        cache = self._get_cache()
        token_count = 0
        max_tokens = query.tokens_to_generate if query else 256

        print(f"[Tail] Starting generation loop — max {max_tokens} tokens")

        while True:
            msg_type, hidden = self.receive_hidden()

            if msg_type == MSG_STOP:
                print(f"[Tail] Received MSG_STOP — breaking")
                break

            hidden = hidden.to(self.model.device)
            print(f"[Tail] Token {token_count}: received hidden shape {hidden.shape}, forwarding...")
            self.model.pass_counter["i"] = token_count

            token, cache = self.model.forward(hidden, cache)
            print(f"[Tail] Token {token_count}: forward done, token_id={token.item()}")

            # check EOS
            eos_ids = self.model.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]

            if token.item() in eos_ids:
                print(f"[Tail] EOS token detected — sending EOS to master")
                self.send_eos()
                break

            self.model.generated_ids.append(token.item())
            print(f"[Tail] Token {token_count}: sending token to master...")
            self.send_token(token)
            print(f"[Tail] Token {token_count}: token sent")
            token_count += 1

            if token_count >= max_tokens:
                print(f"[Tail] Max tokens reached — sending EOS to master")
                self.send_eos()
                break

        self._set_cache(cache)
        print(f"[Tail] Generation complete — {token_count} tokens")

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
