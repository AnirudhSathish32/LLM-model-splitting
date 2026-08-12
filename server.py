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
        out.append({
            "id": sid,
            "title": (first[:60] if first else "New conversation"),
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


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"deleted": session_id}


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
