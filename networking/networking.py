"""
Shared networking utilities for distributed LLM inference.
Imported by machine_a.py, machine_b.py, validation_a.py, validation_b.py
"""

import socket
import io
import time
import torch
import struct
from config import (
    MACHINE_A_TAILSCALE_IP,
    TAILSCALE_PORT,
    DEVICE,
    MSG_TOKEN,
    MSG_EOS,
    MSG_LAYER,
    MSG_TTFT,
    MSG_STOP,
    DEBUG,
    MSG_FIRST_PASS,
    MSG_NEXT_PASS
)

def logging(msg):
    print(msg)


# ================================================================
# LOW LEVEL
# ================================================================

def from_bytes(payload):
    return torch.load(io.BytesIO(payload), map_location=DEVICE)

def to_bytes(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()




# ================================================================
# CONNECTION SETUP
# ================================================================

def setup_machine_a_conn():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Create server socket
    # AF_INET = IPv4 addressing
    # SOCK_STREAM means TCP
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Allows for the reuse of the port immediately after the program exits
    server_socket.bind(("0.0.0.0", TAILSCALE_PORT))
    # Listen across all network interfaces on the TAILSCALE port
    server_socket.listen(1)
    # backlog size = 1, waiting for incoming connections
    print(f"Machine A listening on port {TAILSCALE_PORT}...")
    conn, addr = server_socket.accept()
    # when Machine B connects we return conn and addr
    print(f"Machine B connected from {addr}")
    return server_socket, conn

def setup_machine_b_conn(retries=20, delay=3):
    print(f"Machine B connecting to {MACHINE_A_TAILSCALE_IP}:{TAILSCALE_PORT}")
    for attempt in range(1, retries + 1):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Create client socket
            # AF_INET = IPv4 addressing
            # SOCK_STREAM means TCP
            client_socket.connect((MACHINE_A_TAILSCALE_IP, TAILSCALE_PORT))
            # Attempts TCP handshake
            print(f"Connected to Machine A on attempt {attempt}")
            return client_socket
        except ConnectionRefusedError:
            print(f"Attempt {attempt}/{retries} — Machine A not ready, retrying in {delay}s...")
            client_socket.close()
            time.sleep(delay)
    raise ConnectionError("Could not connect to Machine A")

# ================================================================
# FILE TRANSFER
# ================================================================

def send_msg_file(conn, msg_type, filepath):
    with open(filepath, "rb") as f:
        send_message(conn, msg_type, f.read())
    logging(f"sent file {filepath}")

def receive_msg_file(conn, expected_msg_type, save_path):
    _, payload = read_message(conn, expect = expected_msg_type)
    with open(save_path, "wb") as f:
        f.write(payload)
    logging(f"saved file {save_path} ({len(payload)} bytes)")


def send_handoff(conn, msg_type, hidden, position_embeddings, position_ids=None):
    cos, sin = position_embeddings
    if position_ids is not None:
        payload = to_bytes({
            "hidden": hidden,
            "cos": cos,
            "sin": sin,
            "position_ids":position_ids,
        })
    else: 
        payload = to_bytes({
            "hidden": hidden,
            "cos": cos,
            "sin": sin,
        })
    send_message(conn, msg_type, payload)

def receive_handoff(conn, expect=None):
    _, payload = read_message(conn, expect=expect)
    pkg = from_bytes(payload)
    if expect == MSG_FIRST_PASS:
        hidden = pkg["hidden"]
        position_embeddings = (pkg["cos"], pkg["sin"])
        position_ids = pkg["position_ids"]
        return hidden, position_embeddings, position_ids
    if expect == MSG_NEXT_PASS:
        hidden = pkg["hidden"]
        position_embeddings = (pkg["cos"], pkg["sin"])
        return hidden, position_embeddings
    



# ================================================================
# TOKEN COMMUNICATION
# ================================================================

def send_stop(conn): 
    send_message(conn, MSG_STOP)

def send_ttft(conn, ttft): 
    """
    Send Time-To-First-Token as an 8-byte float
    """
    send_message(conn, MSG_TTFT, struct.pack(">d", ttft))
    logging(f"sent TTFT: {ttft:.4f}s")

def receive_ttft(conn):
    """
    Receive TTFT from Machine A
    """

    _, payload = read_message(conn, expect=MSG_TTFT)
    ttft = struct.unpack(">d", payload)[0]
    logging(f"received TTFT: {ttft:.4f}s")

def send_token(conn, token):
    send_message(conn, MSG_TOKEN, to_bytes(token))

def send_eos(conn):
    send_message(conn, MSG_EOS)

def receive_response(conn):
    msg_type, payload = read_message(conn)
    if msg_type == MSG_EOS:
        return "eos", None
    if msg_type == MSG_TOKEN:
        return "token", from_bytes(payload)
    raise ValueError(f"Unexpected msg {msg_type}")

# ================================================================
# LAYER OUTPUT EXCHANGE
# ================================================================


def send_layers(conn, layers): 
    send_message(conn, MSG_LAYER, to_bytes(layers))
    

def receive_layers(conn):
    _, payload = read_message(conn, expect=MSG_LAYER)
    return from_bytes(payload)





