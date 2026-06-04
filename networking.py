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
    MSG_STOP
)

# ================================================================
# CONNECTION SETUP
# ================================================================

def send_to_machine_b(conn, filepath):
    with open(filepath, "rb") as f:
        # Open file in binary read mode
        # tensor files contain raw serialized bytes so text would corrupt the data
        data = f.read()
        # load file into memory
    conn.sendall(len(data).to_bytes(8, byteorder="big"))
    # len(data).tobytes(8) = let the first 8 bytes = the file length
    # byteorder = big = send the most siginificant byte first
    # we are telling the receiver how much data is coming 
    conn.sendall(data)
    # sending the actual data
    print(f"Sent {filepath} ({len(data)} bytes)")

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
# LOW LEVEL
# ================================================================

def read_message(conn): 
    msg_type = read_TCP_data(conn, 1)[0] 
    length = int.from_bytes(read_TCP_data(conn, 8), "big") 
    payload = read_TCP_data(conn, length) 
    return msg_type, payload

def read_TCP_data(conn, length):
    """
        helper function

        conn = TCP socket connection between Machine A and B brokered by Tailscale
        length = exact number of bytes expected in the incoming data

        returns data in binary format
    
    """
    data = b""
    # empty bytes buffer, this is raw binary data

    while len(data) < length:
        # we loop until we have enough bytes collected
        packet = conn.recv(length - len(data))
        # the packet = length needed - length of data currently being processed 
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
        # add packet binaries to data
    return data


# ================================================================
# FILE TRANSFER
# ================================================================

def send_msg_file(conn, msg_type, filepath):
    conn.sendall(msg_type.to_bytes(1, "big"))
    send_to_machine_b(conn, filepath)


def receive_msg_file(conn, expected_msg_type, save_path):
    msg_type = read_TCP_data(conn, 1)[0]
    if msg_type != expected_msg_type:
        raise ValueError(
            f"Expected msg {expected_msg_type}, got {msg_type}"
        )
    receive_file(conn, save_path)

def receive_file(conn, save_path):
    
    length = int.from_bytes(read_TCP_data(conn, 8), byteorder="big")
    # read exactly the first 8 bytes which contain the file size
    # int.from_bytes = turn bytes back into numbers

    data = read_TCP_data(conn, length)
    # read the payload
    
    with open(save_path, "wb") as f:
    # open destination file in binary write mode
        f.write(data)
        # write the data
    print(f"File saved to {save_path},({length}) bytes...")
# ================================================================
# TOKEN COMMUNICATION
# ================================================================

def send_stop(conn): 
    conn.sendall(bytes([MSG_STOP]))
    print("Sent STOP signal to Machine B")

def send_ttft(conn, ttft): 
    """
    Send Time-To-First-Token as an 8-byte float
    """

    payload = struct.pack(">d", ttft)
    # >d:
    # > = big endian
    # d = double precision float (8 bytes)

    conn.sendall(bytes([MSG_TTFT]))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)

    print(f"Sent TTFT: {ttft:.4f}s")

def receive_ttft(conn):
    """
    Receive TTFT from Machine A
    """

    msg_type = read_TCP_data(conn, 1)[0]

    if msg_type != MSG_TTFT:
        raise ValueError(f"Expected MSG_TTFT, got {msg_type}")

    length = int.from_bytes(read_TCP_data(conn, 8), "big")

    payload = read_TCP_data(conn, length)

    ttft = struct.unpack(">d", payload)[0]

    print(f"Received TTFT from Machine A: {ttft:.4f}s")

    return ttft

def send_token(conn, token):
    buffer = io.BytesIO()
    torch.save(token, buffer)
    payload = buffer.getvalue()
    conn.sendall(MSG_TOKEN.to_bytes(1, byteorder="big"))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)

def send_eos(conn):
    conn.sendall(MSG_EOS.to_bytes(1, byteorder="big"))
    conn.sendall((0).to_bytes(8, byteorder="big"))

# ================================================================
# LAYER OUTPUT EXCHANGE
# ================================================================


def send_layers(conn, layers): 
    buffer = io.BytesIO()
    torch.save(layers, buffer)
    payload = buffer.getvalue()
    conn.sendall(MSG_LAYER.to_bytes(1, byteorder="big"))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)
    

def receive_layers(conn):
    msg_type, payload = read_message(conn)
    if msg_type != MSG_LAYER:
        raise ValueError(
            f"Expected MSG_LAYER, got {msg_type}"
        )
    return torch.load(io.BytesIO(payload), map_location=DEVICE)





