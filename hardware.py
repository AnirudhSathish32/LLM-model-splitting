"""
hardware.py

Pure functions for hardware profiling and pipeline construction.
No instance state, no networking, no model loading.

build_pipeline is the core function: given benchmark scores from
all participating machines, it computes the optimal throughput-
proportional layer split and returns the pipeline list that goes
into SharedConfig.
"""

import os
from transformers import AutoConfig


def bytes_per_layer(config, dtype_bytes=2):
    """
    Rough per-decoder-layer weight footprint for a Llama-style model.

    Accounts for:
      - Attention: Q, K, V, O projections (K/V sized for GQA)
      - MLP: gate, up, down projections

    Does NOT account for KV cache (grows at runtime) or activations.
    The safety margin in build_pipeline covers that headroom.
    """
    h    = config.hidden_size
    i    = config.intermediate_size
    n_q  = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head = h // n_q

    attn = h * h + 2 * (h * n_kv * head) + h * h
    mlp  = 3 * h * i

    return (attn + mlp) * dtype_bytes


def build_pipeline(benchmarks, model_path, dtype_bytes=2, safety=0.8):
    """
    Given benchmark results from all participating machines, compute
    the optimal layer split and return the pipeline list for SharedConfig.

    Parameters:
        benchmarks: list of dicts, each from benchmark_machine.load_benchmark
                    plus an "ip" field added by the initiator during collection.
                    [
                        {"ip": "100.0.0.1", "layer_time_s": 0.0008, "vram_free": ..., ...},
                        {"ip": "100.0.0.2", "layer_time_s": 0.0020, "vram_free": ..., ...},
                        ...
                    ]
        model_path: path to the HuggingFace model (for reading layer count + dims)
        dtype_bytes: bytes per parameter (2 for float16/bfloat16, 4 for float32)
        safety: fraction of free memory to actually use (0.8 = 80%)
                reserves 20% for KV cache growth, activations, OS overhead

    Returns:
        pipeline: list of dicts, ordered by chain position
                  [{"ip": ..., "role": ..., "layers": [start, end], "compute": {...}}, ...]
    """
    config = AutoConfig.from_pretrained(model_path)
    n_layers = config.num_hidden_layers
    per_layer = bytes_per_layer(config, dtype_bytes)

    # ──────────────────────────────────────────────────────────
    # Step 1: Compute throughput and memory capacity for each peer
    # ──────────────────────────────────────────────────────────

    for b in benchmarks:
        b["throughput"] = 1.0 / b["layer_time_s"] if b["layer_time_s"] > 0 else float("inf")

        has_gpu = b.get("gpu_name") is not None and b.get("vram_free", 0) > 0
        usable_mem = (b["vram_free"] if has_gpu else b["ram_free"]) * safety

        b["max_layers"] = max(0, int(usable_mem // per_layer) - 1)

    # ──────────────────────────────────────────────────────────
    # Step 2: Feasibility check — can the model fit at all?
    # ──────────────────────────────────────────────────────────

    total_capacity = sum(b["max_layers"] for b in benchmarks)
    if total_capacity < n_layers:
        raise RuntimeError(
            f"Combined capacity ({total_capacity} layers) < model ({n_layers} layers). "
            f"Not enough hardware to run {os.path.basename(model_path)}. "
            f"Need {n_layers - total_capacity} more layers worth of memory."
        )

    # ──────────────────────────────────────────────────────────
    # Step 3: Filter out machines too weak to hold any layers
    # ──────────────────────────────────────────────────────────

    viable = [b for b in benchmarks if b["max_layers"] >= 1]
    excluded = [b for b in benchmarks if b["max_layers"] < 1]

    for b in excluded:
        print(f"  Excluding {b['ip']}: cannot hold even 1 layer "
              f"({b.get('vram_free', 0) / 1e9:.1f}GB VRAM, "
              f"{b.get('ram_free', 0) / 1e9:.1f}GB RAM)")

    if not viable:
        raise RuntimeError("No machines can hold any layers.")

    # ──────────────────────────────────────────────────────────
    # Step 4: Sort by throughput — fastest machine becomes master
    # ──────────────────────────────────────────────────────────

    viable.sort(key=lambda b: b["throughput"], reverse=True)

    # ──────────────────────────────────────────────────────────
    # Step 5: Compute proportional layer counts
    # ──────────────────────────────────────────────────────────

    total_throughput = sum(b["throughput"] for b in viable)

    layer_cursor = 0
    pipeline = []

    for idx, b in enumerate(viable):
        is_last = (idx == len(viable) - 1)
        remaining = n_layers - layer_cursor

        if is_last:
            # last machine gets everything remaining — avoids rounding gaps
            count = remaining
        else:
            # proportional share, rounded to nearest integer
            share = b["throughput"] / total_throughput
            count = max(1, round(n_layers * share))

            # guard: never assign more than what's left minus 1 per remaining machine
            # this reserves at least 1 layer for every machine still in the loop
            machines_after = len(viable) - idx - 1
            count = min(count, remaining - machines_after)

            # clamp: never exceed memory capacity
            count = min(count, b["max_layers"])

            # floor: at least 1 layer per machine in the pipeline
            count = max(1, count)

        # assign role based on position in sorted list
        if idx == 0:
            role = "master"
        elif is_last:
            role = "tail"
        else:
            role = "worker"

        # assign contiguous layer range
        layer_start = layer_cursor
        layer_end = layer_cursor + count - 1

        pipeline.append({
            "ip":     b["ip"],
            "role":   role,
            "layers": [layer_start, layer_end],
            "compute": {
                "layer_time_s":   b["layer_time_s"],
                "throughput":     b["throughput"],
                "max_layers":     b["max_layers"],
                "assigned":       count,
                "expected_time":  count * b["layer_time_s"],
                "device":         b.get("device", "unknown"),
                "gpu_name":       b.get("gpu_name"),
            },
        })

        layer_cursor = layer_end + 1

    # ──────────────────────────────────────────────────────────
    # Step 6: Verify full layer coverage
    # ──────────────────────────────────────────────────────────

    total_assigned = sum(entry["compute"]["assigned"] for entry in pipeline)
    if total_assigned != n_layers:
        raise RuntimeError(
            f"Split error: assigned {total_assigned} layers but model has {n_layers}. "
            f"This is a bug in the proportional split logic."
        )

    # ──────────────────────────────────────────────────────────
    # Step 7: Print the split summary
    # ──────────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"PIPELINE SPLIT: {os.path.basename(model_path)} ({n_layers} layers)")
    print(f"{'='*72}")
    print(f"{'Role':<8} {'IP':<18} {'Layers':<14} {'Count':<7} "
          f"{'ms/layer':<10} {'Stage ms':<10} {'Device'}")
    print(f"{'─'*72}")

    stage_times = []
    for entry in pipeline:
        c = entry["compute"]
        stage_ms = c["expected_time"] * 1000
        stage_times.append(stage_ms)
        layer_range = f"{entry['layers'][0]}..{entry['layers'][1]}"
        device_str = c["gpu_name"] or c["device"]
        print(f"{entry['role']:<8} {entry['ip']:<18} {layer_range:<14} "
              f"{c['assigned']:<7} {c['layer_time_s']*1000:<10.3f} "
              f"{stage_ms:<10.2f} {device_str}")

    print(f"{'─'*72}")
    bottleneck = max(stage_times)
    fastest = min(stage_times)
    print(f"Bottleneck stage:  {bottleneck:.2f} ms")
    print(f"Fastest stage:     {fastest:.2f} ms")
    print(f"Balance ratio:     {fastest/bottleneck:.2f} (1.0 = perfect)")
    print(f"Est. per-token:    {bottleneck:.2f} ms (limited by slowest stage)")
    print(f"{'='*72}\n")

    return pipeline

    
