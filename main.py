"""
main.py

Send a query through the full system: Tailscale discovery → benchmark
collection → pipeline build → config distribution → generation → response.

Prerequisites:
    1. Both machines running:  python daemon.py
    2. Both machines have benchmarks:  python benchmark.py llama-3b
    3. Tailscale connected (tailscale status shows both peers online)

Usage:
    python main.py "What is the capital of France?"
    python main.py                  # uses default prompt
"""

import sys
import torch

from config import LocalConfig
from user_query import UserQuery, send_query
from session import Session, SessionManager


class SimpleSessionStore:
    """Minimal in-memory session store for testing."""
    def __init__(self):
        self._store = {}

    def save(self, session):
        self._store[session.session_id] = session

    def load(self, session_id):
        return self._store.get(session_id)

    def list_all(self):
        return list(self._store.keys())

    def delete(self, session_id):
        self._store.pop(session_id, None)


def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    model_name = "llama-3b"
    tokens = 20

    local = LocalConfig.load()

    query = UserQuery(
        prompt=prompt,
        model_name=model_name,
        session_id="cli-session",
        tokens_to_generate=tokens,
        dtype=torch.float16,
    )

    store = SimpleSessionStore()
    session_manager = SessionManager(store=store)

    print(f"Prompt: {prompt}")
    print(f"Model:  {model_name}")
    print(f"Tokens: {tokens}\n")

    response = send_query(query, local, session_manager)

    print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()