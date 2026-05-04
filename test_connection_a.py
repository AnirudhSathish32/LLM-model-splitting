import socket
import io
import torch
import os

TAILSCALE_PORT = 65432

def recv_all(conn, length):
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
    return data

def setup_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", TAILSCALE_PORT))
    server_socket.listen(1)
    print(f"Machine A listening on port {TAILSCALE_PORT}...")
    conn, addr = server_socket.accept()
    print(f"Machine B connected from {addr}")
    return server_socket, conn

def send_file(conn, filepath):
    # Read file as raw bytes
    with open(filepath, "rb") as f:
        data = f.read()
    # Send length first then data
    conn.sendall(len(data).to_bytes(8, byteorder="big"))
    conn.sendall(data)
    print(f"Sent {filepath} ({len(data)} bytes)")

if __name__ == "__main__":
    # Create a dummy tensor and save it
    dummy_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dummy_tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    torch.save(dummy_tensor, "./dummy.pt")
    torch.save(dummy_tensor2, "./dummy2.pt")
    print(f"Created dummy tensor: {dummy_tensor}")
    print(f"Created dummy tensor: {dummy_tensor2}")

    # Connect
    server_socket, conn = setup_server()

    # Send the file
    send_file(conn, "./dummy.pt")
    send_file(conn, "./dummy2.pt")

    # Wait for Machine B to confirm receipt
    confirm = recv_all(conn, 4)
    print(f"Machine B confirmed: {confirm.decode()}")
    

    conn.close()
    server_socket.close()
    print("Machine A done")