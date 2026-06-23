# hardware.py
import torch, psutil

def get_hardware_profile():
    """Each machine measures its own capacity. Sent during handshake."""
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory
        vram_free  = vram_total - torch.cuda.memory_allocated(0)
    else:
        vram_total = vram_free = 0
    return {
        "has_cuda":    has_cuda,
        "vram_free":   vram_free,                       # bytes available for weights+cache
        "ram_free":    psutil.virtual_memory().available,
        "device":      "cuda" if has_cuda else "cpu",
        # a crude compute proxy; refine later with a micro-benchmark
        "compute_score": (torch.cuda.get_device_properties(0).multi_processor_count
                          if has_cuda else psutil.cpu_count(logical=False) or 1),
    }


def bytes_per_layer(config, dtype_bytes=2):
    """
    Rough per-decoder-layer weight footprint for a Llama-style model.
    Attention: q,k,v,o projections. MLP: gate, up, down. GQA shrinks k,v.
    """
    h      = config.hidden_size
    i      = config.intermediate_size
    n_q    = config.num_attention_heads
    n_kv   = config.num_key_value_heads
    head   = h // n_q
    attn   = h*h + 2*(h * n_kv*head) + h*h          # q + (k,v gqa) + o
    mlp    = 3 * h * i                               # gate, up, down
    return (attn + mlp) * dtype_bytes


def compute_optimal_split(config, profile_a, profile_b, safety=0.8):
    """
    Decide how many layers Machine A holds (0..split-1); B holds the rest.

    Strategy: give each machine a layer count proportional to its usable memory,
    capped so neither exceeds its budget (weights + headroom for KV cache + the
    embedding/norm/head extras). Returns `split` = stopping_layer.
    """
    n_layers = config.num_hidden_layers
    per_layer = bytes_per_layer(config)

    # A also holds embed_tokens; B holds norm + lm_head. Approximate as ~1 layer each.
    budget_a = profile_a["vram_free"] * safety if profile_a["has_cuda"] else profile_a["ram_free"] * safety
    budget_b = profile_b["vram_free"] * safety if profile_b["has_cuda"] else profile_b["ram_free"] * safety

    max_layers_a = max(0, int(budget_a // per_layer) - 1)   # -1 for embeddings
    max_layers_b = max(0, int(budget_b // per_layer) - 1)   # -1 for norm+head

    # Degenerate cases — bail to single-machine
    if max_layers_a + max_layers_b < n_layers:
        raise RuntimeError(
            f"Combined capacity ({max_layers_a + max_layers_b} layers) "
            f"< model size ({n_layers}). Model won't fit on this pair."
        )
    if max_layers_b < 4:
        return "machine_a_only"     # B too weak to be worth the round-trip

    # Within the feasible range, balance by compute so neither stage is the bottleneck.
    total_compute = profile_a["compute_score"] + profile_b["compute_score"]
    split = round(n_layers * profile_a["compute_score"] / total_compute)

    # Clamp into the memory-feasible window.
    split = max(n_layers - max_layers_b, min(split, max_layers_a))
    split = max(1, min(split, n_layers - 1))    # both machines keep ≥1 layer
    return split