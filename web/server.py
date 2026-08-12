"""
web/server.py

FastAPI layer over the distributed inference orchestrator.

The daemon does the inference; this process is the initiator — it owns
the session store and calls send_query() exactly the way main.py does.
Tokens stream back over the control connection and are relayed to the
browser as newline-delimited JSON.

Started by launch.py, not run directly.
"""

import os
import sys
import json
import queue
import threading
import traceback

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LocalConfig
from session import SessionManager
from user_query import UserQuery, send_query, clear_pipeline, get_pipeline_info
from logbuffer import BUFFER
from config import MSG_LOG_REQ, MSG_LOG_RESP
from networking.protocol import send_message, read_message

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Distributed Inference")

local = LocalConfig.load()
session_manager = SessionManager()

# One query at a time per session — a follow-up needs the previous answer.
_session_locks = {}
_locks_guard = threading.Lock()


def _session_lock(session_id):
    with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


# ── Static ───────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


# ── Models ───────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    """Models that have been split into per-layer files."""
    path = getattr(local, "layers_path", None) or "./layers"
    try:
        names = sorted(
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        )
    except FileNotFoundError:
        names = []
    return {"models": names}


# ── Pipeline topology ────────────────────────────────────────

@app.get("/api/pipeline")
def pipeline(model: str):
    info = get_pipeline_info(model)
    return {"pipeline": info, "self_ip": local.tailscale_ip}


# ── Sessions ─────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions():
    out = []
    for sid in session_manager.list_sessions():
        try:
            s = session_manager.get_or_create(sid)
        except Exception:
            continue
        first = next(
            (m["content"] for m in s.messages if m["role"] == "user"), None
        )
        title = getattr(s, "title", None) or (
            first[:60] if first else "New conversation"
        )
        out.append({
            "id": sid,
            "title": title,
            "custom_title": bool(getattr(s, "title", None)),
            "turns": sum(1 for m in s.messages if m["role"] == "user"),
            "last_active": getattr(s, "last_active", 0),
        })
    out.sort(key=lambda x: x["last_active"], reverse=True)
    return {"sessions": out}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = session_manager.get_or_create(session_id)
    return {
        "id": session_id,
        "messages": [m for m in s.messages if m["role"] != "system"],
    }


class NewSession(BaseModel):
    id: str


@app.post("/api/sessions")
def create_session(body: NewSession):
    s = session_manager.get_or_create(body.id)
    session_manager.save_session(s)
    return {"id": body.id}


class RenameSession(BaseModel):
    title: str


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, body: RenameSession):
    title = session_manager.set_title(session_id, body.title)
    return {"id": session_id, "title": title}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"deleted": session_id}




# ── Logs ─────────────────────────────────────────────────────

def _fetch_remote_logs(ip, since, daemon_port=65433, timeout=3.0):
    """Ask another node for its recent output over the daemon port."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, daemon_port))
        send_message(sock, MSG_LOG_REQ, str(since).encode("utf-8"))
        msg_type, payload = read_message(sock)
        if msg_type != MSG_LOG_RESP:
            return {"error": f"unexpected reply {msg_type}"}
        return json.loads(payload.decode("utf-8"))
    finally:
        sock.close()


@app.get("/api/logs")
def logs(ip: str = "", since: int = 0):
    """
    Recent console output. Omit ip (or pass this machine's) to read the
    local buffer directly; any other ip is fetched from that node.
    """
    if not ip or ip == local.tailscale_ip:
        return {
            "ip": local.tailscale_ip,
            "device": local.device,
            "local": True,
            "lines": BUFFER.read(since=since),
            "latest_seq": BUFFER.latest_seq(),
        }

    try:
        data = _fetch_remote_logs(ip, since)
        data["local"] = False
        return data
    except Exception as e:
        return {
            "ip": ip,
            "local": False,
            "lines": [],
            "latest_seq": since,
            "error": f"{type(e).__name__}: {e}",
        }


@app.post("/api/logs/clear")
def clear_logs():
    BUFFER.clear()
    return {"cleared": True}


# ── Settings ─────────────────────────────────────────────────

# Fields the running process picks up on the next query vs. those
# baked into loaded models and open sockets at startup.
LIVE_FIELDS = {"overhead", "debug", "model_path", "layers_path"}
RESTART_FIELDS = {"device", "tailscale_ip"}


@app.get("/api/settings")
def get_settings():
    return {
        "settings": {
            "device": local.device,
            "tailscale_ip": local.tailscale_ip,
            "model_path": local.model_path,
            "layers_path": local.layers_path,
            "overhead": local.overhead,
            "debug": local.debug,
        },
        "restart_fields": sorted(RESTART_FIELDS),
        "config_path": LocalConfig.CONFIG_PATH,
    }


class Settings(BaseModel):
    device: str
    tailscale_ip: str
    model_path: str
    layers_path: str
    overhead: float
    debug: bool


@app.put("/api/settings")
def put_settings(body: Settings):
    """
    Write settings to disk and update the running orchestrator where
    that is safe. Reports which changes need a restart to take effect.
    """
    if not 0.0 <= body.overhead < 1.0:
        return JSONResponse(
            status_code=400,
            content={"error": "Memory reserve must be between 0 and 1 "
                              "(0.2 reserves 20%)."},
        )

    incoming = body.model_dump()
    changed = [k for k, v in incoming.items() if getattr(local, k) != v]

    # Mutate in place so anything holding this instance sees the update.
    for k, v in incoming.items():
        setattr(local, k, v)

    try:
        local.save()
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"Could not write config: {e}"})

    needs_restart = sorted(set(changed) & RESTART_FIELDS)
    return {
        "saved": True,
        "changed": changed,
        "needs_restart": needs_restart,
    }


# ── Chat (streaming) ─────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    model: str
    prompt: str
    tokens: int = 200


@app.post("/api/chat")
def chat(body: ChatRequest):
    """
    Runs send_query on a worker thread. on_token pushes deltas onto a
    queue; this generator drains it and emits NDJSON lines the browser
    reads incrementally.
    """
    q = queue.Queue()

    def on_token(delta):
        q.put({"type": "token", "text": delta})

    def worker():
        lock = _session_lock(body.session_id)
        with lock:
            try:
                query = UserQuery(
                    prompt=body.prompt,
                    model_name=body.model,
                    session_id=body.session_id,
                    tokens_to_generate=body.tokens,
                    dtype=torch.float16,
                )
                response = send_query(
                    query, local, session_manager, on_token=on_token
                )
                q.put({"type": "done", "text": response})
            except Exception as e:
                traceback.print_exc()
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/clear-pipeline")
def clear():
    clear_pipeline()
    return {"cleared": True}
