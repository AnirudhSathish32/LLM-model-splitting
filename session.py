import uuid
import time

class Session:
    def __init__(self, tokenizer, session_id=None, system_prompt=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.tokenizer = tokenizer
        self.messages = []
        self.cached_token_count = 0
        self.created_at = time.time()
        self.last_active = time.time()

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content):
        self.messages.append({"role":"user","content":content})