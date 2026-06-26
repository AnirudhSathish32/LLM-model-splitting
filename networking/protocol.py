"""
Protocol for overall networking 
1 Length Message, 8 byte length of payload, payload

Protocol for serialization of bytes

"""

import torch, io

def logging(msg):
    print(msg)

def read_message(conn, expect=None): 
    msg_type = read_TCP_data(conn, 1)[0] 
    length = int.from_bytes(read_TCP_data(conn, 8), "big") 
    payload = read_TCP_data(conn, length)
    if expect is not None and msg_type != expect:
        raise ValueError(f"Expected msg {expect}, got {msg_type}") 
    return msg_type, payload

def send_message(conn, msg_type, payload=b""):
    conn.sendall(msg_type.to_bytes(1, "big"))
    conn.sendall(len(payload).to_bytes(8, "big"))
    conn.sendall(payload)
    logging(f"sent msg_type={msg_type} ({len(payload)} bytes)")

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

def from_bytes(payload):
    return torch.load(io.BytesIO(payload), map_location=DEVICE)

def to_bytes(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()
