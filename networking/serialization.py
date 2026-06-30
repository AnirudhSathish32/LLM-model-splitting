import torch, io
from dataclasses import asdict

def tensor_from_bytes(payload, device="cpu"):
    return torch.load(io.BytesIO(payload), map_location=device)

def tensor_to_bytes(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()

def to_bytes(obj):
    """Serialize a dataclass instance to bytes via torch.save."""
    buf = io.BytesIO()
    torch.save(asdict(obj), buf)
    return buf.getvalue()
 
 
def from_bytes(cls, data):
    """Deserialize bytes back into a dataclass instance."""
    d = torch.load(io.BytesIO(data), weights_only=False)
    return cls(**d)

def serialize_config_query(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    payload = buf.getvalue()
    return payload