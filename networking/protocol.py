"""
Wire protocol for distributed inference.

Format: 1-byte msg_type | 8-byte payload length (big-endian) | payload bytes
"""


def read_message(conn, expect=None):
    """Read one framed message. Optionally assert msg_type."""
    msg_type = read_TCP_data(conn, 1)[0]
    length = int.from_bytes(read_TCP_data(conn, 8), "big")
    payload = read_TCP_data(conn, length)
    if expect is not None and msg_type != expect:
        raise ValueError(f"Expected msg {expect}, got {msg_type}")
    return msg_type, payload


def send_message(conn, msg_type, payload=b""):
    """Send one framed message."""
    conn.sendall(msg_type.to_bytes(1, "big"))
    conn.sendall(len(payload).to_bytes(8, "big"))
    conn.sendall(payload)


def read_TCP_data(conn, length):
    """Read exactly `length` bytes from a TCP connection."""
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
    return data
