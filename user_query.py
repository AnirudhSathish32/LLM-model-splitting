from dataclasses import dataclass, field
import torch
import socket
import threading
import time
import os

from hardware import build_pipeline
from config import (SharedConfig, LocalConfig, MSG_CONFIG, MSG_READY, MSG_START,
                    MSG_RESPONSE, MSG_QUERY, MSG_QUERY_FAIL, MSG_TOKEN_STREAM)
from networking.protocol import send_message, read_message
from networking.serialization import to_bytes, serialize_config_query

import zlib

def _model_port(model_name, base=60000, span=2000):
    """
    Deterministic port per model so concurrent multi-model pipelines
    don't collide. Uses crc32 (not Python hash(), which is salted
    per-process). Both orchestrators compute the same port for the
    same model independently — no coordination needed.

    Maps into [base, base+span) = [60000, 62000), well within the
    valid TCP range (0-65535) and clear of common service ports.
    """
    offset = zlib.crc32(model_name.encode()) % span
    return base + offset


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


# Timing records collected from the daemon during the last query.
# The harness reads these after send_query returns.
_last_timing_records = []


def _read_until_response(conn, on_token=None):
    """
    Read from the master's control connection, forwarding MSG_TOKEN_STREAM
    deltas to on_token, until a terminal message arrives.
    Timing records from the daemon are collected into _last_timing_records.
    Returns (msg_type, payload) for the terminal message.
    """
    global _last_timing_records
    import json as _json

    while True:
        msg_type, payload = read_message(conn)
        if msg_type == MSG_TOKEN_STREAM:
            text = payload.decode("utf-8")
            # Timing records arrive as JSON with type=timing
            if text.startswith('{"type": "timing"'):
                try:
                    rec = _json.loads(text)
                    rec.pop("type", None)
                    _last_timing_records.append(rec)
                except Exception:
                    pass
            elif on_token:
                try:
                    on_token(text)
                except Exception:
                    pass
            continue
        return msg_type, payload


def get_last_timing_records():
    """Return timing records from the most recent distributed query, then clear."""
    global _last_timing_records
    out = list(_last_timing_records)
    _last_timing_records = []
    return out


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


def get_pipeline_info(model_name):
    """
    Report the currently cached pipeline for a model, for UI display.
    Returns None if no pipeline is cached yet (cold start pending).
    """
    cached = _cached_pipelines.get(model_name)
    if cached is None:
        return None
    if cached.get("local_only"):
        return {"mode": "local", "stages": []}
    shared = cached.get("shared")
    if shared is None:
        return None
    return {
        "mode": "distributed",
        "port": shared.port,
        "stages": [
            {
                "ip": e["ip"],
                "role": e["role"],
                "layer_start": e["layers"][0],
                "layer_end": e["layers"][1],
                "count": e["layers"][1] - e["layers"][0] + 1,
            }
            for e in shared.pipeline
        ],
    }

# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION — the initiator's flow from query to inference
# ═══════════════════════════════════════════════════════════════

def send_query(query, local: LocalConfig, session_manager, daemon_port=65433, on_token=None):
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
                response = _run_local(query, local, session, on_token=on_token)
            else:
                response = _try_warm_query(query, cached, daemon_port, on_token=on_token)

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
        response = _run_local(query, local, session, on_token=on_token)
 
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
            msg_type, payload = _read_until_response(master_conn, on_token)
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
# LOCAL MODEL ASSEMBLY — rebuild a full model from per-layer files
# ═══════════════════════════════════════════════════════════════

class MissingLayerFiles(RuntimeError):
    """Raised when layers/<model>/ can't produce a complete model."""


def _check_layer_files(model_dir, layers_dir, model_name):
    """
    Verify everything needed to assemble the full model is present before
    loading any of it, so a missing file fails immediately with a useful
    message instead of part-way through a multi-GB load.

    Returns (config, expected_layer_count, needs_head).
    """
    import os
    from transformers import AutoConfig

    if not os.path.isdir(layers_dir):
        raise MissingLayerFiles(
            f"No layer files for '{model_name}'.\n"
            f"  Expected directory: {layers_dir}\n"
            f"  Create it with:     python create_layer_files.py {model_name}"
        )

    # config.json / tokenizer live in models/, not layers/
    required_json = [
        "config.json", "tokenizer_config.json", "special_tokens_map.json",
    ]
    missing_json = [
        f for f in required_json
        if not os.path.exists(os.path.join(model_dir, f))
    ]
    if not any(os.path.exists(os.path.join(model_dir, f))
               for f in ("tokenizer.json", "tokenizer.model")):
        missing_json.append("tokenizer.json (or tokenizer.model)")

    if missing_json:
        raise MissingLayerFiles(
            f"'{model_name}' is missing tokenizer/config files in {model_dir}:\n"
            + "".join(f"    {f}\n" for f in missing_json)
            + "  These are small JSON files and must be kept even after "
              "splitting the weights."
        )

    config = AutoConfig.from_pretrained(model_dir)
    n_layers = config.num_hidden_layers

    # Tied embeddings mean lm_head shares embed_tokens' weight, so a
    # head file may legitimately be absent or empty.
    needs_head = not getattr(config, "tie_word_embeddings", False)

    expected = [f"layer_{i}.safetensors" for i in range(n_layers)]
    expected += ["embed_tokens.safetensors", "norm.safetensors"]
    if needs_head:
        expected.append("head.safetensors")

    missing = [f for f in expected
               if not os.path.exists(os.path.join(layers_dir, f))]

    if missing:
        shown = missing[:8]
        more = f"    ...and {len(missing) - 8} more\n" if len(missing) > 8 else ""
        raise MissingLayerFiles(
            f"'{model_name}' is missing {len(missing)} of {len(expected)} "
            f"layer files in {layers_dir}:\n"
            + "".join(f"    {f}\n" for f in shown) + more
            + f"  Re-create them with: python create_layer_files.py {model_name}\n"
              f"  (that step needs the full weights in {model_dir})"
        )

    return config, n_layers, needs_head


def _has_full_weights(model_dir):
    """Whether the original multi-GB weight files are still on disk."""
    import os, glob
    if not os.path.isdir(model_dir):
        return False
    return bool(glob.glob(os.path.join(model_dir, "*.safetensors"))
                or glob.glob(os.path.join(model_dir, "*.bin")))


def _assemble_local_model(model_dir, layers_dir, model_name, dtype, device):
    """
    Rebuild a complete model from per-layer safetensors.

    The distributed path loads only its own slice; this loads every layer
    plus embeddings, norm, and head, so the single-machine path no longer
    needs the original multi-GB weight files on disk.
    """
    import os
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config, n_layers, needs_head = _check_layer_files(
        model_dir, layers_dir, model_name
    )

    print(f"[Local] Assembling {model_name} from {n_layers} layer files...")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    state = {}
    try:
        state.update(load_file(
            os.path.join(layers_dir, "embed_tokens.safetensors"), device="cpu"))
        for i in range(n_layers):
            state.update(load_file(
                os.path.join(layers_dir, f"layer_{i}.safetensors"), device="cpu"))
        state.update(load_file(
            os.path.join(layers_dir, "norm.safetensors"), device="cpu"))

        head_path = os.path.join(layers_dir, "head.safetensors")
        if os.path.exists(head_path):
            state.update(load_file(head_path, device="cpu"))
    except Exception as e:
        raise MissingLayerFiles(
            f"Could not read layer files for '{model_name}': {e}\n"
            f"  A file may be corrupt or partially written. Re-create with:\n"
            f"    python create_layer_files.py {model_name}"
        ) from e

    # Tied embeddings: lm_head reuses the embedding matrix.
    if not needs_head and "lm_head.weight" not in state:
        embed = state.get("model.embed_tokens.weight")
        if embed is not None:
            state["lm_head.weight"] = embed

    state = {k: v.to(dtype) for k, v in state.items()}

    missing_keys, unexpected = model.load_state_dict(
        state, strict=False, assign=True
    )

    # Any parameter still on the meta device never received a weight.
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta:
        raise MissingLayerFiles(
            f"'{model_name}' assembled with {len(still_meta)} uninitialised "
            f"weights, e.g. {still_meta[:3]}.\n"
            f"  The layer files are incomplete or from a different model.\n"
            f"  Re-create with: python create_layer_files.py {model_name}"
        )

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"[Local] {model_name} assembled on {device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# LOCAL SINGLE-NODE PATH
# ═══════════════════════════════════════════════════════════════
 
def _run_local(query, local, session, on_token=None):
    """
    Run inference on a single machine — no pipeline splitting, no networking.
    Model is cached per model_name so switching models doesn't reload.
    """
    import os
    global _local_models

    model_name = query.model_name
    model_dir = os.path.join(local.model_path, model_name)
    layers_dir = os.path.join(local.layers_path, model_name)

    # Load model on first call for this model, reuse on subsequent calls
    if model_name not in _local_models:
        try:
            model, tokenizer = _assemble_local_model(
                model_dir, layers_dir, model_name, query.dtype, local.device
            )
        except MissingLayerFiles as e:
            # Fall back to the original weights if they happen to still be
            # on disk; otherwise surface the layer-file problem, which is
            # the one the user can actually fix.
            full_weights = _has_full_weights(model_dir)
            if not full_weights:
                raise
            print(f"[Local] Layer files unusable, falling back to full weights")
            print(f"[Local]   ({str(e).splitlines()[0]})")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=query.dtype
            ).to(local.device)
            model.eval()

        _local_models[model_name] = {"model": model, "tokenizer": tokenizer}

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

    import perf_telemetry as _tel
    _t_gen_start = time.time()

    if on_token:
        # Stream deltas as they are produced
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=query.tokens_to_generate,
            do_sample=False,
            streamer=streamer,
        )
        gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        gen_thread.start()

        parts = []
        _tok_idx = 0
        for chunk in streamer:
            parts.append(chunk)
            if _tel.is_enabled():
                _tel.record(
                    session=query.session_id,
                    token_index=_tok_idx,
                    phase="prefill" if _tok_idx == 0 else "decode",
                    input_tokens=input_len if _tok_idx == 0 else 1,
                    context_tokens=input_len + _tok_idx,
                    hidden_bytes=0,
                    master_compute_ms=0,
                    roundtrip_ms=0,
                    step_ms=0,
                    master_layers=0,
                    path="local",
                )
                _tok_idx += 1
            try:
                on_token(chunk)
            except Exception:
                pass
        gen_thread.join()
        response = "".join(parts)
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=query.tokens_to_generate,
                do_sample=False,
            )
        new_tokens = outputs[0][input_len:]
        n_generated = len(new_tokens)
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Record per-token telemetry from wall time
        if _tel.is_enabled() and n_generated > 0:
            _wall = (time.time() - _t_gen_start) * 1000.0
            _per_tok = _wall / n_generated
            for _i in range(n_generated):
                _tel.record(
                    session=query.session_id,
                    token_index=_i,
                    phase="prefill" if _i == 0 else "decode",
                    input_tokens=input_len if _i == 0 else 1,
                    context_tokens=input_len + _i,
                    hidden_bytes=0,
                    master_compute_ms=round(_per_tok, 4),
                    roundtrip_ms=0.0,
                    step_ms=round(_per_tok, 4),
                    master_layers=0,
                    path="local",
                )
 
    return response

# ═══════════════════════════════════════════════════════════════
# WARM PATH — MSG_QUERY to existing peers
# ═══════════════════════════════════════════════════════════════
 
def _try_warm_query(query, cached, daemon_port, on_token=None):
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
    msg_type, resp_payload = _read_until_response(master_conn, on_token)
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