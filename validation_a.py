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
from config import (
    MODEL_PATH,
    STOPPING_LAYER,
    PROMPT,
    TOKENS_TO_GENERATE,
    DEVICE
)
    

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
    
def validate_all_layers(full_outputs, split_outputs, full_model, split_model, cos_threshold=0.99, tolerance=1e-2,):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=-1)

    print(f"\n{'='*65}")
    print(f"LAYER VALIDATION: MACHINE A  — Full ({len(full_model.model.layers)}) Layers vs Split ({len(split_model.model.layers)} Layers)")
    print(f"{'='*65}")
    print(f"{'Layer':<8} {'Mean Diff':<14} {'Cos Sim':<12} {'Match'}")
    print(f"{'-'*65}")

    all_match = True
    for idx in sorted(full_outputs.keys()):
        if idx not in split_outputs:
            print(f"{idx:<8} NOT CAPTURED")
            continue

        full_h  = full_outputs[idx].float().to(DEVICE)
        split_h = split_outputs[idx].float().to(DEVICE)

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
    # ---- Split generation ----
    print("\n[1] Running split generation...")
    server_socket, conn = machine_a.setup_machine_a_conn()
    model, inputs, tokenizer = machine_a.setup_model_a(STOPPING_LAYER, MODEL_PATH, PROMPT)
    split_monitor = ResourceMonitor()
    split_monitor.start()
    split_start = time.time()
    split_response, all_layer_outputs, split_ttft = machine_a.run_machine_a(TOKENS_TO_GENERATE, STOPPING_LAYER, tokenizer, inputs, model, conn)
    split_time  = time.time() - split_start
    split_monitor.stop()
    split_stats = split_monitor.summary()

    # ---- Full generation ----
    print("\n[2] Running full generation...")
    full_monitor = ResourceMonitor()
    full_monitor.start()
    full_start  = time.time()
    full_result = generation.default_generation(MODEL_PATH, PROMPT, STOPPING_LAYER, TOKENS_TO_GENERATE)
    full_ttft = full_result["ttft"]
    full_response = full_result["response"]
    full_model = full_result["model"]
    full_time   = time.time() - full_start
    full_monitor.stop()
    full_stats  = full_monitor.summary()

    # ---- Validate ----
    validate_all_layers(full_result["layer_outputs"], all_layer_outputs,full_model, model)

    # ---- Response Comparison ----

    print(f"\n{'='*55}")
    print("RESPONSE COMPARISON: MACHINE A")
    print(f"{'='*55}")
    print(f"MODEL: {os.path.basename(MODEL_PATH)}")
    print(f"Split Query: {PROMPT}")
    print(f"Split Response: {split_response}")
    print(f"{'-'*55}")
    print(f"MODEL: {os.path.basename(MODEL_PATH)}")
    print(f"Full Query: {PROMPT}")
    print(f"Full Response: {full_response}")
    print(f"{'-'*55}")

    # ---- Resource comparison ----
    print(f"\n{'='*55}")
    print("RESOURCE COMPARISON: MACHINE A")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'Full':<15} {'Split':<15}")
    print(f"{'-'*55}")
    print(f"{'Time (s)':<25} {full_time:<15.2f} {split_time:<15.2f}")
    print(f"{'Time to First Token (s)':<25} {full_ttft:<15.2f} {split_ttft:<15.2f}")
    print(f"{'CPU peak (%)':<25} {full_stats['cpu_peak']:<15.1f} {split_stats['cpu_peak']:<15.1f}")
    print(f"{'RAM peak (GB)':<25} {full_stats['ram_peak_gb']:<15.2f} {split_stats['ram_peak_gb']:<15.2f}")
    print(f"{'GPU peak (GB)':<25} {full_stats['gpu_peak_gb']:<15.2f} {split_stats['gpu_peak_gb']:<15.2f}")
    print(f"{'='*55}")

    conn.close()
    server_socket.close()
