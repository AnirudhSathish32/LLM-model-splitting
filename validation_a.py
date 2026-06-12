import torch
import time
import os
import psutil
import threading
 
import machine_a
import generation
from config import (
    MODEL_PATH,
    STOPPING_LAYER,
    PROMPT,
    TOKENS_TO_GENERATE,
    DEVICE,
    DEBUG,
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
        cpus = [r["cpu"]    for r in self.records] or [0.0]
        rams = [r["ram_gb"] for r in self.records] or [0.0]
        gpus = [r["gpu_gb"] for r in self.records] or [0.0]
        return {"cpu_peak": max(cpus), "ram_peak_gb": max(rams), "gpu_peak_gb": max(gpus)}
 
 
def validate_layer_histories(full_history, split_history, cos_threshold=0.999, label="MACHINE A"):
    """
    Compare two (pass, layer) -> {"hidden": tensor} grids via cosine similarity.
    Both grids are keyed by GLOBAL layer index, so the full-model baseline and the
    merged split history line up directly.
    """
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
 
    full_keys  = {k for k, v in full_history.items()  if isinstance(v, dict) and "hidden" in v}
    split_keys = {k for k, v in split_history.items() if isinstance(v, dict) and "hidden" in v}
    common     = sorted(full_keys & split_keys)
    only_full  = sorted(full_keys  - split_keys)
    only_split = sorted(split_keys - full_keys)
 
    print(f"\n{'='*74}")
    print(f"LAYER-HISTORY VALIDATION: {label}   (match if cos > {cos_threshold})")
    print(f"{'='*74}")
    print(f"{'Pass':<6}{'Layer':<8}{'Shape':<24}{'Cos Sim':<12}{'Match'}")
    print(f"{'-'*74}")
 
    all_match = True
    per_pass  = {}
    for (p, l) in common:
        fh = full_history[(p, l)]["hidden"].float().to(DEVICE)
        sh = split_history[(p, l)]["hidden"].float().to(DEVICE)
 
        if fh.shape != sh.shape:
            print(f"{p:<6}{l:<8}{str(tuple(fh.shape)) + ' vs ' + str(tuple(sh.shape)):<24}{'SHAPE MISMATCH':<12}✗")
            all_match = False
            per_pass.setdefault(p, []).append(False)
            continue
 
        cos_sim = cos_fn(
            fh.reshape(-1, fh.shape[-1]),
            sh.reshape(-1, sh.shape[-1]),
        ).mean().item()
        match = cos_sim > cos_threshold
        all_match = all_match and match
        per_pass.setdefault(p, []).append(match)
        print(f"{p:<6}{l:<8}{str(tuple(fh.shape)):<24}{cos_sim:<12.6f}{'✓' if match else '✗'}")
 
    print(f"{'-'*74}")
    print("Per-pass summary:")
    for p in sorted(per_pass):
        res = per_pass[p]
        print(f"  Pass {p:<4} {sum(res)}/{len(res)} layers match")
 
    if only_full:
        print(f"\n{len(only_full)} keys in FULL but not SPLIT (e.g. {only_full[:5]})")
    if only_split:
        print(f"{len(only_split)} keys in SPLIT but not FULL (e.g. {only_split[:5]})")
 
    print(f"\nOverall: {'ALL MATCH' if all_match else 'MISMATCH'}  ({len(common)} layer-passes compared)")
    print(f"{'='*74}\n")
    return all_match
 
 
if __name__ == "__main__":
    if not DEBUG:
        print("WARNING: DEBUG=False — the split run only captures the boundary layer, so the "
              "validation grid will be nearly empty. Set DEBUG=True in config.py to validate.")
 
    # ---- Split generation (Machine A side) ----
    print("\n[1] Running split generation...")
    server_socket, conn = machine_a.setup_machine_a_conn()
    model, inputs, tokenizer = machine_a.setup_model_a(STOPPING_LAYER, MODEL_PATH, PROMPT)
 
    split_monitor = ResourceMonitor(); split_monitor.start()
    split_start = time.time()
    split_response, split_history, split_ttft = machine_a.run_machine_a(
        TOKENS_TO_GENERATE, STOPPING_LAYER, tokenizer, inputs, model, conn
    )
    split_time = time.time() - split_start
    split_monitor.stop(); split_stats = split_monitor.summary()
 
    # ---- Full generation (baseline) ----
    print("\n[2] Running full generation...")
    full_monitor = ResourceMonitor(); full_monitor.start()
    full_start = time.time()
    full_result = generation.default_generation(MODEL_PATH, PROMPT, STOPPING_LAYER, TOKENS_TO_GENERATE)
    full_time = time.time() - full_start
    full_monitor.stop(); full_stats = full_monitor.summary()
 
    full_history  = full_result["layer_history"]
    full_response = full_result["response"]
    full_model    = full_result["model"]
    full_ttft     = full_result["ttft"]
 
    # ---- Validate the grids against each other ----
    validate_layer_histories(full_history, split_history, label="MACHINE A")
 
    # ---- Response comparison (the decode-path check) ----
    print(f"{'='*60}")
    print("RESPONSE COMPARISON: MACHINE A")
    print(f"{'='*60}")
    print(f"MODEL: {os.path.basename(MODEL_PATH)}   |   PROMPT: {PROMPT}")
    print(f"Split ({len(model.model.layers)} layers held) response: {split_response}")
    print(f"Full  ({len(full_model.model.layers)} layers) response: {full_response}")
    print(f"Responses identical: {split_response == full_response}")
    print(f"{'-'*60}")
 
    # ---- Resource comparison ----
    print(f"\n{'='*60}")
    print("RESOURCE COMPARISON: MACHINE A")
    print(f"{'='*60}")
    print(f"{'Metric':<26}{'Full':<16}{'Split':<16}")
    print(f"{'-'*60}")
    print(f"{'Time (s)':<26}{full_time:<16.2f}{split_time:<16.2f}")
    print(f"{'TTFT (s)':<26}{(full_ttft or 0):<16.2f}{(split_ttft or 0):<16.2f}")
    print(f"{'CPU peak (%)':<26}{full_stats['cpu_peak']:<16.1f}{split_stats['cpu_peak']:<16.1f}")
    print(f"{'RAM peak (GB)':<26}{full_stats['ram_peak_gb']:<16.2f}{split_stats['ram_peak_gb']:<16.2f}")
    print(f"{'GPU peak (GB)':<26}{full_stats['gpu_peak_gb']:<16.2f}{split_stats['gpu_peak_gb']:<16.2f}")
    print(f"{'='*60}")
 
    conn.close()
    server_socket.close()
