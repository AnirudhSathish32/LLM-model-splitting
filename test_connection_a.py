import socket
import io
import torch
import os

TAILSCALE_PORT = 65432

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

def setup_machine_a():
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

def send_file(conn, filepath):
    
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