import torch
import time
import os
import psutil
import threading
 
import machine_b
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
 
 
def validate_layer_histories(full_history, split_history, cos_threshold=0.999, label="MACHINE B"):
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

def print_timing_comparison(full_history, split_history, label=""):
    """
    For each pass, average the per-layer dur across all layers and compare
    full-model vs split. Prefill (pass 0) is naturally slower — it processes
    seq_len tokens. Decode passes (1+) process one token each and are the
    meaningful steady-state comparison.
    """
    # group dur values by pass for each history
    def group_by_pass(history):
        passes = {}
        for (p, l), v in history.items():
            if isinstance(v, dict) and "dur" in v:
                passes.setdefault(p, []).append(v["dur"])
        return passes

    full_by_pass  = group_by_pass(full_history)
    split_by_pass = group_by_pass(split_history)
    all_passes    = sorted(full_by_pass.keys() | split_by_pass.keys())

    print(f"\n{'='*72}")
    print(f"PER-PASS TIMING COMPARISON  {label}")
    print(f"  dur = avg wall-clock seconds across all layers in that pass")
    print(f"  pass 0 = prefill (seq_len tokens);  pass 1+ = decode (1 token each)")
    print(f"{'='*72}")
    print(f"{'Pass':<6} {'Type':<10} {'Full (s)':<12} {'Split (s)':<12} {'Speedup':>8}")
    print(f"{'-'*72}")

    prefill_full  = []
    prefill_split = []
    decode_full   = []
    decode_split  = []

    for p in all_passes:
        f_durs = full_by_pass.get(p, [])
        s_durs = split_by_pass.get(p, [])
        f_avg  = sum(f_durs)  / len(f_durs)  if f_durs  else float("nan")
        s_avg  = sum(s_durs)  / len(s_durs)  if s_durs  else float("nan")
        speedup = f_avg / s_avg if s_avg > 0 else float("nan")
        kind    = "prefill" if p == 0 else "decode"

        print(f"{p:<6} {kind:<10} {f_avg:<12.4f} {s_avg:<12.4f} {speedup:>7.2f}x")

        if p == 0:
            prefill_full.append(f_avg);  prefill_split.append(s_avg)
        else:
            decode_full.append(f_avg);   decode_split.append(s_avg)

    print(f"{'-'*72}")

    # summary row for decode steady-state
    if decode_full and decode_split:
        df_avg = sum(decode_full)  / len(decode_full)
        ds_avg = sum(decode_split) / len(decode_split)
        print(f"{'avg':<6} {'decode':<10} {df_avg:<12.4f} {ds_avg:<12.4f} {df_avg/ds_avg:>7.2f}x")

    if prefill_full and prefill_split:
        pf = prefill_full[0]; ps = prefill_split[0]
        print(f"{'0':<6} {'prefill':<10} {pf:<12.4f} {ps:<12.4f} {pf/ps:>7.2f}x  (single sample)")

    print(f"{'='*72}\n")
 
 
if __name__ == "__main__":
    if not DEBUG:
        print("WARNING: DEBUG=False — the split run only captures the boundary layer, so the "
              "validation grid will be nearly empty. Set DEBUG=True in config.py to validate.")
 
    # ---- Split generation (Machine B side) ----
    print("\n[1] Running split generation...")
    conn = machine_b.setup_machine_b_conn()
    model, tokenizer = machine_b.setup_model_b(STOPPING_LAYER, MODEL_PATH)
 
    split_monitor = ResourceMonitor(); split_monitor.start()
    split_start = time.time()
    split_response, split_history, split_ttft = machine_b.run_machine_b(
        tokenizer, model, STOPPING_LAYER, TOKENS_TO_GENERATE, conn
    )
    split_time = time.time() - split_start
    split_monitor.stop(); split_stats = split_monitor.summary()
 
    # ---- Full generation (baseline) ----
    # NOTE: this loads the FULL model on Machine B. If B is the weaker node this may not
    # fit in memory — run validation primarily from Machine A in that case.
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
    validate_layer_histories(full_history, split_history, label="MACHINE B")
    print_timing_comparison(full_history, split_history, label="MACHINE B")
 
    # ---- Response comparison (the decode-path check) ----
    print(f"{'='*60}")
    print("RESPONSE COMPARISON: MACHINE B")
    print(f"{'='*60}")
    print(f"MODEL: {os.path.basename(MODEL_PATH)}   |   PROMPT: {PROMPT}")
    print(f"Split ({len(model.model.layers)} layers held) response: {split_response}")
    print(f"Full  ({len(full_model.model.layers)} layers) response: {full_response}")
    print(f"Responses identical: {split_response == full_response}")
    print(f"{'-'*60}")
 
    # ---- Resource comparison ----
    print(f"\n{'='*60}")
    print("RESOURCE COMPARISON: MACHINE B")
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
