import socket
import io
import torch
import os
import time

MACHINE_A_TAILSCALE_IP = "100.74.100.92"
TAILSCALE_PORT = 65432

def recv_all(conn, length):
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
    return data

def setup_client(retries=20, delay=3):
    print(f"Machine B connecting to {MACHINE_A_TAILSCALE_IP}:{TAILSCALE_PORT}")
    for attempt in range(1, retries + 1):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((MACHINE_A_TAILSCALE_IP, TAILSCALE_PORT))
            print(f"Connected to Machine A on attempt {attempt}")
            return client_socket
        except ConnectionRefusedError:
            print(f"Attempt {attempt}/{retries} — Machine A not ready, retrying in {delay}s...")
            client_socket.close()
            time.sleep(delay)
    raise ConnectionError("Could not connect to Machine A")

def receive_file(conn, save_path):
    # Receive length first
    length = int.from_bytes(recv_all(conn, 8), byteorder="big")
    print(f"Receiving {length} bytes...")
    
    # Receive file bytes
    data = recv_all(conn, length)
    
    # Write to disk
    with open(save_path, "wb") as f:
        f.write(data)
    print(f"File saved to {save_path}")

if __name__ == "__main__":
    # Connect to Machine A
    conn = setup_client()

    # Receive the file
    print("getting file 1")
    receive_file(conn, "./received_dummy.pt")
    print("getting file 2")
    receive_file(conn, "./received_dummy2.pt")

    # Load and print the tensor
    tensor = torch.load("./received_dummy.pt")
    tensor2 = torch.load("./received_dummy2.pt")
    print(f"Received tensor: {tensor}")
    print(f"Shape: {tensor.shape}")

    # Send confirmation back to Machine A
    conn.sendall(b"done")
    print("Confirmation sent to Machine A")

    conn.close()
    print("Machine B done")