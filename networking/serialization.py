import torch, io, struct
from dataclasses import asdict

# ═══════════════════════════════════════════════════════════════
# TENSOR WIRE FORMAT — raw bytes, no pickle
#
# Header: [dtype:1][ndim:1][shape:4*ndim] + raw data
# Cross-platform, no torch.save overhead (~1KB saved per message),
# readable by any language that can parse bytes.
# ═══════════════════════════════════════════════════════════════
 
_DTYPE_TO_ID = {
    torch.float16:  0,
    torch.bfloat16: 1,
    torch.float32:  2,
    torch.float64:  3,
    torch.int32:    4,
    torch.int64:    5,
    torch.int8:     6,
    torch.uint8:    7,
}
 
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}
 
 
def tensor_to_bytes(tensor):
    """Serialize a tensor to raw bytes: header + contiguous data."""
    tensor = tensor.detach().contiguous().cpu()
    dtype_id = _DTYPE_TO_ID[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
 
    # header: 1 byte dtype + 1 byte ndim + 4 bytes per dim
    header = struct.pack(f">BB{ndim}I", dtype_id, ndim, *shape)
    return header + tensor.numpy().tobytes()
 
 
def tensor_from_bytes(payload, device="cpu"):
    """Deserialize raw bytes back into a tensor on the target device."""
    # read header
    dtype_id = payload[0]
    ndim = payload[1]
    header_size = 2 + 4 * ndim
 
    shape = struct.unpack(f">{ndim}I", payload[2:header_size])
    dtype = _ID_TO_DTYPE[dtype_id]
 
    # read data
    data = payload[header_size:]
    tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    return tensor.to(device)

def to_bytes(obj):
    """Serialize a dataclass instance to bytes via torch.save."""
    buf = io.BytesIO()
    torch.save(asdict(obj), buf)
    return buf.getvalue()

def from_bytes(cls, data):
    """Deserialize bytes back into a dataclass instance."""
    d = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return cls(**d)

def serialize_config_query(obj):
    """Serialize a dict (e.g. bundled shared+query) to bytes."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()
