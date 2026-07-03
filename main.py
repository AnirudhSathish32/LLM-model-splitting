"""
main.py

Interactive multi-turn chat through the distributed inference pipeline.

First query: cold path (discover peers, load models, connect chain).
Subsequent queries: warm path (reuse peers, models, connections, caches).
Conversation history persists to ./sessions/<session_id>.json.

Prerequisites:
    1. Both machines running:  python networking/daemon.py
    2. Both machines have benchmarks:  python benchmark.py llama-3b

Usage:
    python main.py                            # interactive mode
    python main.py "What is 2+2?"             # one-shot mode
"""

import sys
import torch

from config import LocalConfig
from user_query import UserQuery, send_query, clear_pipeline
from session import SessionManager


MODEL_NAME = "llama-8b"
TOKENS = 50
SESSION_ID = "interactive"


def main():
    local = LocalConfig.load()
    session_manager = SessionManager()

    # ── One-shot mode ────────────────────────────────────
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        query = UserQuery(
            prompt=prompt,
            model_name=MODEL_NAME,
            session_id=SESSION_ID,
            tokens_to_generate=TOKENS,
            dtype=torch.float16,
        )
        response = send_query(query, local, session_manager)
        print(f"\nAssistant: {response}")
        return

    # ── Interactive mode ─────────────────────────────────
    print(f"Model: {MODEL_NAME} | Tokens: {TOKENS} | Session: {SESSION_ID}")
    print("Commands: quit, clear, history\n")

    # Show previous conversation if resuming
    existing = session_manager.get_or_create(SESSION_ID)
    if existing.messages:
        n = len([m for m in existing.messages if m["role"] == "user"])
        print(f"Resuming session ({n} previous turns):")
        for msg in existing.messages:
            role = msg["role"]
            text = msg["content"][:100]
            if role == "user":
                print(f"  You: {text}")
            elif role == "assistant":
                print(f"  Assistant: {text}")
        print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            break

        if prompt.lower() == "clear":
            session_manager.delete_session(SESSION_ID)
            clear_pipeline()
            print("Conversation and pipeline cleared.\n")
            continue

        if prompt.lower() == "history":
            session = session_manager.get_or_create(SESSION_ID)
            if not session.messages:
                print("(empty)\n")
            else:
                for msg in session.messages:
                    role = msg["role"]
                    print(f"  [{role}] {msg['content'][:120]}")
                print()
            continue

        query = UserQuery(
            prompt=prompt,
            model_name=MODEL_NAME,
            session_id=SESSION_ID,
            tokens_to_generate=TOKENS,
            dtype=torch.float16,
        )

        try:
            response = send_query(query, local, session_manager)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()