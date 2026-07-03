import uuid
import time
import json
import os

from config import LocalConfig


class Session:
    def __init__(self, session_id=None, system_prompt=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages = []
        self.cached_token_count = 0
        self.created_at = time.time()
        self.last_active = time.time()

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        self.last_active = time.time()

    def add_assistant_message(self, content, model_name):
        self.messages.append({
            "role": "assistant",
            "content": content,
            "model": model_name,
        })
        self.last_active = time.time()

    def invalidate_cache(self):
        self.cached_token_count = 0

    @property
    def system_prompt(self):
        if self.messages and self.messages[0]["role"] == "system":
            return self.messages[0]["content"]
        return None


class SessionManager:
    def __init__(self, sessions_path=None, default_system_prompt=None, max_sessions=10):
        self.sessions_path = sessions_path or LocalConfig.SESSION_PATH
        self.default_system_prompt = default_system_prompt
        self.max_sessions = max_sessions
        self.sessions = {}       # in-memory cache of active sessions

        os.makedirs(self.sessions_path, exist_ok=True)

    def _file_path(self, session_id):
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.sessions_path, f"{safe_id}.json")

    def get_or_create(self, session_id, system_prompt=None):
        # check in-memory first
        if session_id in self.sessions:
            return self.sessions[session_id]

        # try loading from disk (previous app run)
        session = self._load_from_disk(session_id)
        if session:
            self.sessions[session_id] = session
            return session

        # truly new conversation
        session = Session(
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
        )
        self.sessions[session_id] = session
        self._save_to_disk(session)

        # evict oldest if over limit
        if len(self.sessions) > self.max_sessions:
            self._evict_oldest()

        return session

    def save_session(self, session):
        """Write-through: memory + disk."""
        self.sessions[session.session_id] = session
        self._save_to_disk(session)

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)
        path = self._file_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        return session_id

    def list_sessions(self):
        """List all session IDs from disk."""
        ids = []
        for f in os.listdir(self.sessions_path):
            if f.endswith(".json"):
                ids.append(f[:-5])
        return ids

    def _save_to_disk(self, session):
        data = {
            "session_id": session.session_id,
            "messages": session.messages,
            "cached_token_count": session.cached_token_count,
            "created_at": session.created_at,
            "last_active": session.last_active,
        }
        with open(self._file_path(session.session_id), "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self, session_id):
        path = self._file_path(session_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        session = Session(session_id=data["session_id"])
        session.messages = data["messages"]
        session.cached_token_count = data.get("cached_token_count", 0)
        session.created_at = data.get("created_at", time.time())
        session.last_active = data.get("last_active", time.time())
        return session

    def _evict_oldest(self):
        oldest_id = min(
            self.sessions,
            key=lambda sid: self.sessions[sid].last_active,
        )
        self.delete_session(oldest_id)
        return oldest_id