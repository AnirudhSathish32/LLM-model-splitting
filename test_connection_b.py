import socket
import io
import torch
import os
import time

MACHINE_A_TAILSCALE_IP = "100.74.100.92"
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

def setup_machine_b(retries=20, delay=3):
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

def receive_file(conn, save_path):
    
    length = int.from_bytes(read_TCP_data(conn, 8), byteorder="big")
    # read exactly the first 8 bytes which contain the file size
    # int.from_bytes = turn bytes back into numbers

    print(f"Receiving {length} bytes...")
    
    data = read_TCP_data(conn, length)
    # read the payload
    
    with open(save_path, "wb") as f:
    # open destination file in binary write mode
        f.write(data)
        # write the data
    print(f"File saved to {save_path}")

if __name__ == "__main__":
    # Connect to Machine A
    conn = setup_machine_b()

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