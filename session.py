import uuid
import time

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
    def __init__(self, store, default_system_prompt=None, max_sessions=10):
        self.store = store
        self.default_system_prompt = default_system_prompt
        self.max_sessions = max_sessions
        self.sessions = {}       # in-memory cache of active sessions

    def get_or_create(self, session_id, system_prompt=None):
        # check in-memory first
        if session_id in self.sessions:
            return self.sessions[session_id]

        # try loading from disk (previous app run)
        session = self.store.load(session_id)
        if session:
            self.sessions[session_id] = session
            return session

        # truly new conversation
        session = Session(
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
        )
        self.sessions[session_id] = session
        self.store.save(session)

        # evict oldest if over limit
        if len(self.sessions) > self.max_sessions:
            evicted_id = self._evict_oldest()
            # TODO: tell peers to purge caches for evicted_id

        return session

    def save_session(self, session):
        """Write-through: memory + disk."""
        self.sessions[session.session_id] = session
        self.store.save(session)

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)
        self.store.delete(session_id)
        return session_id    # caller tells peers to purge caches

    def list_sessions(self):
        """For the frontend sidebar."""
        return self.store.list_all()

    def _evict_oldest(self):
        oldest_id = min(
            self.sessions,
            key=lambda sid: self.sessions[sid].last_active,
        )
        self.sessions.pop(oldest_id)
        self.store.delete(oldest_id)
        return oldest_id