"""
launch.py

One entry point per machine. Starts the inference daemon in a background
thread, serves the chat UI, and opens a browser.

Run this on every machine that should participate. The daemon makes the
machine available as a pipeline stage; the UI lets you send queries from
whichever machine you happen to be sitting at.

    python launch.py                # daemon + UI, opens browser
    python launch.py --no-browser   # daemon + UI, no browser
    python launch.py --daemon-only  # headless participant, no UI
    python launch.py --port 8100    # UI on a different port

Prerequisite: run benchmark.py <model> once per machine so the
orchestrator knows how fast this machine is.
"""

import os
import sys
import time
import argparse
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tee console output into a ring buffer the UI can read. Must happen
# before the daemon or server produce any output.
from logbuffer import install as install_log_capture
install_log_capture()


def start_daemon(daemon_port):
    """Run the inference daemon in a background thread."""
    from networking.daemon import Daemon
    from config import LocalConfig

    d = Daemon(LocalConfig.load(), port=daemon_port)

    t = threading.Thread(target=d.start, daemon=True)
    t.start()
    print(f"[Launch] Daemon listening on {daemon_port}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="UI port")
    ap.add_argument("--daemon-port", type=int, default=65433)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--daemon-only", action="store_true")
    args = ap.parse_args()

    from config import LocalConfig
    local = LocalConfig.load()
    print(f"[Launch] This machine: {local.tailscale_ip} ({local.device})")

    start_daemon(args.daemon_port)

    if args.daemon_only:
        print("[Launch] Daemon-only mode — Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Launch] Stopped")
        return

    import uvicorn
    from web.server import app

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print(f"[Launch] UI at {url}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
