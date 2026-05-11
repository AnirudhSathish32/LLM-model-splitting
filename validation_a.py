import torch
import time
import os
import io
import psutil
import threading
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import machine_a
import generation
    

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.records  = []
        self.running  = False
        self._thread  = None

    def start(self):
        self.running = True
        self.records = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join()

    def _run(self):
        while self.running:
            self.records.append({
                "cpu":    psutil.cpu_percent(interval=None),
                "ram_gb": psutil.Process(os.getpid()).memory_info().rss / 1e9,
                "gpu_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            })
            time.sleep(self.interval)

    def summary(self):
        cpus = [r["cpu"]    for r in self.records]
        rams = [r["ram_gb"] for r in self.records]
        gpus = [r["gpu_gb"] for r in self.records]
        return {
            "cpu_peak":    max(cpus),
            "ram_peak_gb": max(rams),
            "gpu_peak_gb": max(gpus),
        }
    
def validate_all_layers(full_outputs, split_outputs, cos_threshold=0.99, tolerance=1e-2):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=-1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print("LAYER VALIDATION — Full vs Split (all 28 layers)")
    print(f"{'='*65}")
    print(f"{'Layer':<8} {'Mean Diff':<14} {'Cos Sim':<12} {'Match'}")
    print(f"{'-'*65}")

    all_match = True
    for idx in sorted(full_outputs.keys()):
        if idx not in split_outputs:
            print(f"{idx:<8} NOT CAPTURED")
            continue

        full_h  = full_outputs[idx].float().to(device)
        split_h = split_outputs[idx].float().to(device)

        max_diff  = (full_h - split_h).abs().max().item()
        mean_diff = (full_h - split_h).abs().mean().item()
        cos_sim   = cos_sim_fn(
            full_h.reshape(-1,  full_h.shape[-1]),
            split_h.reshape(-1, split_h.shape[-1])
        ).mean().item()
        match = cos_sim > cos_threshold
        if not match:
            all_match = False

        print(f"{idx:<8} {mean_diff:<14.6f} {cos_sim:<12.6f} {'✓' if match else '✗'}")

    print(f"{'-'*65}")
    print(f"All layers match: {all_match}")
    return all_match

if __name__ == "__main__":
    model_path = "./llama-3b"
    stopping_layer = 14
    starting_layer = stopping_layer + 1
    prompt = "Hello world"
    tokens_to_generate = 50 

    # ---- Split generation ----
    print("\n[2] Running split generation...")
    server_socket, conn = machine_a.setup_machine_a_conn()
    model, inputs, tokenizer = machine_a.setup_model_a(stopping_layer, model_path, prompt)
    split_monitor = ResourceMonitor()
    split_monitor.start()
    split_start = time.time()
    response, all_layer_outputs = machine_a.run_machine_a(tokens_to_generate, stopping_layer, tokenizer, inputs, model, conn)
    print(f"Response: {response}")
    split_time  = time.time() - split_start
    split_monitor.stop()
    split_stats = split_monitor.summary()

    # ---- Full generation ----
    print("\n[1] Running full generation...")
    full_monitor = ResourceMonitor()
    full_monitor.start()
    full_start  = time.time()
    full_result = generation.default_generation(model_path, prompt, stopping_layer, tokens_to_generate)
    full_time   = time.time() - full_start
    full_monitor.stop()
    full_stats  = full_monitor.summary()

    # ---- Validate ----
    validate_all_layers(full_result["layer_outputs"], all_layer_outputs)

    # ---- Resource comparison ----
    print(f"\n{'='*55}")
    print("RESOURCE COMPARISON")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'Full':<15} {'Split':<15}")
    print(f"{'-'*55}")
    print(f"{'Time (s)':<25} {full_time:<15.2f} {split_time:<15.2f}")
    print(f"{'CPU peak (%)':<25} {full_stats['cpu_peak']:<15.1f} {split_stats['cpu_peak']:<15.1f}")
    print(f"{'RAM peak (GB)':<25} {full_stats['ram_peak_gb']:<15.2f} {split_stats['ram_peak_gb']:<15.2f}")
    print(f"{'GPU peak (GB)':<25} {full_stats['gpu_peak_gb']:<15.2f} {split_stats['gpu_peak_gb']:<15.2f}")
    print(f"{'='*55}")

    conn.close()
    server_socket.close()
