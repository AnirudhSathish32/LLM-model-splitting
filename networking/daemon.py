import os, sys, json, socket, threading, time

from config import LocalConfig, MSG_PROFILE

class Daemon:

    def __init__(self, local_config=None, port=65433):
        self.local = local_config or LocalConfig.load()
        self.port = port
        self.running = False

    def start(self):
        """Bind and listen. Blocks forever."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(5)
        self.running = True
 
        print(f"[Daemon] Listening on port {self.port}")
        print(f"[Daemon] IP: {self.local.tailscale_ip}")
        print(f"[Daemon] Layers dir: {self.local.layers_dir}")
        print(f"[Daemon] Models dir: {self.local.model_dir}")
 
        try:
            while self.running:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr),
                    daemon=True,
                )
                thread.start()
        except KeyboardInterrupt:
            print("\n[Daemon] Shutting down")
        finally:
            server.close()


    def _handle_connection(self, conn, addr):
        """Handle one incoming request. Runs in its own thread."""
        try:
            msg_type, payload = read_message(conn)
 
            if msg_type == MSG_PING:
                self._handle_ping(conn)
 
            elif msg_type == MSG_BENCHMARK_REQ:
                model_name = payload.decode("utf-8")
                self._handle_benchmark_request(conn, model_name)
 
            else:
                print(f"[Daemon] Unknown message type {msg_type} from {addr}")
 
        except ConnectionError as e:
            print(f"[Daemon] Connection error from {addr}: {e}")
        except Exception as e:
            print(f"[Daemon] Error handling {addr}: {e}")
        finally:
            conn.close()