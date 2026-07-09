"""
main.py

Interactive multi-turn chat with concurrent query support.
Queries dispatch in background threads — you can type the next
prompt before the previous response arrives.

Usage:
    python main.py                            # interactive mode
    python main.py "What is 2+2?"             # one-shot mode
"""

import sys
import torch
import threading

from config import LocalConfig
from user_query import UserQuery, send_query, clear_pipeline
from session import SessionManager


MODEL_NAME = "llama-3b"
TOKENS = 50
SESSION_ID = "testing thread"

_query_counter = 0
_query_lock = threading.Lock()


def dispatch_query(prompt, local, session_manager, query_num):
    """Run a query in a background thread. Prints response when done."""
    # Each concurrent query gets its own session to avoid interleaving corruption
    session_id = f"{SESSION_ID}-{query_num}"

    query = UserQuery(
        prompt=prompt,
        model_name=MODEL_NAME,
        session_id=session_id,
        tokens_to_generate=TOKENS,
        dtype=torch.float16,
    )

    try:
        response = send_query(query, local, session_manager)
        print(f"\nAssistant (#{query_num}: {prompt[:40]}): {response}\n", flush=True)
    except Exception as e:
        print(f"\nError (#{query_num}): {e}\n", flush=True)
        import traceback
        traceback.print_exc()


def main():
    global _query_counter

    local = LocalConfig.load()
    session_manager = SessionManager()

    # one-shot mode
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

    # interactive mode
    print(f"Model: {MODEL_NAME} | Tokens: {TOKENS}")
    print("Commands: quit, clear, history")
    print("Type multiple prompts — responses arrive when ready\n")

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
            clear_pipeline()
            print("Pipeline cleared.\n")
            continue

        _query_counter += 1
        t = threading.Thread(
            target=dispatch_query,
            args=(prompt, local, session_manager, _query_counter),
            daemon=True,
        )
        t.start()


if __name__ == "__main__":
    main()
