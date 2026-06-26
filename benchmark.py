import torch, psutil, time, os, sys, json
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors.torch import load_file
from config import LocalConfig, LAYERS_DIR, MODEL_PATH, DTYPE, DEVICE

def benchmark_single_layer(model_path, layers_dir, device, dtype, warmup=3, trials=10):
    """
    Benchmarking for llama-3B and llama-8B
    """
    
    print(f"Benchmarking on device: {device}, dtype: {dtype}")
    print(f"Model: {model_path}")
 
    config = AutoConfig.from_pretrained(model_path)
    model_name = os.path.basename(model_path)
    layer_file = f"{layers_dir}/layer_0.safetensors"
    embed_file = f"{layers_dir}/embed_tokens.safetensors"
 
    if not os.path.exists(layer_file):
        print(f"ERROR: {layer_file} not found.")
        print(f"Run create_layer_files.py first to split the model into per-layer safetensors.")
        sys.exit(1)

     # ---- build a minimal 1-layer model ----
    bench_config = AutoConfig.from_pretrained(model_path)
    bench_config.num_hidden_layers = 1
 
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(bench_config)
 
    state = {}
    state.update(load_file(embed_file, device=device))
    state.update(load_file(layer_file, device=device))
    model.load_state_dict(
        state, 
        strict=False, 
        assign=True
        )
    model.eval()
 
    # ---- build a decode-step input: single token, shape [1, 1, hidden_dim] ----
    hidden_dim = config.hidden_size
    dummy_input = torch.randn(1, 1, hidden_dim, device=device, dtype=dtype)
 
    layer = model.model.layers[0]
    position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    position_embeddings = model.model.rotary_emb(dummy_input, position_ids)
    is_cuda = device.startswith("cuda")

    # ---- warmup: absorb kernel compilation, cuDNN autotuning, allocator setup ----
    print(f"Warmup: {warmup} passes...")
    with torch.no_grad():
        for _ in range(warmup):
            layer(dummy_input, position_embeddings=position_embeddings)
    if is_cuda:
        torch.cuda.synchronize()
 
    # ---- timed trials ----
    print(f"Benchmarking: {trials} passes...")
    times = []
    with torch.no_grad():
        for i in range(trials):
            if is_cuda:
                torch.cuda.synchronize()
 
            start = time.perf_counter()
            layer(dummy_input, position_embeddings=position_embeddings)
 
            if is_cuda:
                torch.cuda.synchronize()
 
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Trial {i+1}: {elapsed*1000:.3f} ms")
 
    # ---- compute stats ----
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    minimum = times[0]
    maximum = times[-1]
 
    # ---- cleanup: free the benchmark model before anything else loads ----
    del model, layer, dummy_input
    if is_cuda:
        torch.cuda.empty_cache()
 
    # ---- memory snapshot ----
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_total = props.total_memory
        vram_free = vram_total - torch.cuda.memory_allocated(0)
    else:
        gpu_name = None
        vram_total = 0
        vram_free = 0
 
    ram_total = psutil.virtual_memory().total
    ram_free = psutil.virtual_memory().available
 
    result = {
        "model_name":      model_name,
        "device":          device,
        "dtype":           str(dtype),
        "gpu_name":        gpu_name,
        "vram_total":      vram_total,
        "vram_free":       vram_free,
        "ram_total":       ram_total,
        "ram_free":        ram_free,
        "layer_time_s":    median,
        "layers_per_sec":  1.0 / median if median > 0 else float("inf"),
        "total_layers":    config.num_hidden_layers,
        "hidden_dim":      hidden_dim,
        "trials":          trials,
        "times_ms": {
            "median": median * 1000,
            "mean":   mean * 1000,
            "min":    minimum * 1000,
            "max":    maximum * 1000,
        },
    }
 
    return result

def save_benchmark(result, output_dir="./benchmark/"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{result['model_name']}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
 
    print(f"\nBenchmark saved to: {filepath}")
    return filepath

def load_benchmark(model_name, benchmark_dir="./benchmark/"):
    filepath = os.path.join(benchmark_dir, f"{model_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)
    

if __name__ == "__main__":
    """
    if len(sys.argv) < 2:
        print("Usage: python benchmark_machine.py <model_name>")
        print("Example: python benchmark_machine.py llama-8b")
        sys.exit(1)

    model_name = sys.argv[1]
    
    local = LocalConfig.load()

    result = benchmark_single_layer(
        model_path=f"{local.model_dir}/{model_name}",
        layers_dir=local.layers_dir,
        device=local.device,
        dtype=torch.float16,
        warmup=3,
        trials=10,
    )
    """
    result = benchmark_single_layer(
        model_path=MODEL_PATH,
        layers_dir=LAYERS_DIR,
        device=DEVICE,
        dtype=DTYPE,
        warmup=3,
        trials=10,
    )

    save_benchmark(result, output_dir="./benchmark/")

