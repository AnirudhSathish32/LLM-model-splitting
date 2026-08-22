"""
bench_latency.py

Runs the latency experiments for the measurement writeup and dumps a CSV.

Pipeline depth cannot be varied from software — it depends on which
machines are running a daemon. So run this once per hardware setup and
label each run; the plotting script aggregates across the labels.

    # with only this machine running a daemon
    python bench_latency.py --label 1-node

    # start a second machine's daemon, then
    python bench_latency.py --label 2-node

    # start a third, then
    python bench_latency.py --label 3-node

    python plot_results.py results/*.csv

Each configuration writes results/<label>.csv with one row per generated
token, plus results/<label>_summary.csv with per-condition medians.

Options:
    --repeats N       repetitions per condition (default 5)
    --tokens N        tokens to generate per query (default 32)
    --model NAME      model to test (default llama-3b)
    --contexts A,B,C  context lengths to sweep (default 64,256,1024,2048)
    --quick           2 repeats, 2 context lengths — a smoke test
"""

import argparse
import csv
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import perf_telemetry as telemetry
from config import LocalConfig
from session import SessionManager
from user_query import UserQuery, send_query, clear_pipeline, get_pipeline_info

RESULTS_DIR = "results"

# Filler that tokenizes at roughly one token per word, so a target
# context length can be hit without depending on the tokenizer.
_FILLER = (
    "The network carries the hidden state between machines in the pipeline "
    "and each stage adds its own latency to every token that is produced. "
)


def make_prompt(target_tokens, tokenizer=None):
    """Build a prompt of approximately target_tokens tokens."""
    if tokenizer is not None:
        text = ""
        while len(tokenizer(text).input_ids) < target_tokens:
            text += _FILLER
        ids = tokenizer(text).input_ids[:target_tokens]
        return tokenizer.decode(ids)
    words = _FILLER.split()
    return " ".join((words * ((target_tokens // len(words)) + 2))[:target_tokens])


def get_tokenizer(local, model_name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            os.path.join(local.model_path, model_name))
    except Exception as e:
        print(f"  (tokenizer unavailable, using word-count estimate: {e})")
        return None


def describe_pipeline(model_name):
    """Stage count and layer distribution of the pipeline actually used."""
    info = get_pipeline_info(model_name)
    if info is None:
        return {"stages": 0, "topology": "unknown"}
    if info.get("mode") == "local":
        return {"stages": 1, "topology": "single-node"}
    stages = info["stages"]
    return {
        "stages": len(stages),
        "topology": " → ".join(
            f"{s['role']}:{s['count']}L" for s in stages
        ),
    }


def run_condition(local, sm, model_name, context_len, tokens, repeats,
                  label, cold, tokenizer):
    """Run one (context length x path) condition `repeats` times."""
    path = "cold" if cold else "warm"
    print(f"\n  context={context_len:>5}  path={path:<4}  ", end="", flush=True)

    prompt = make_prompt(context_len, tokenizer)
    topology = None

    for rep in range(repeats):
        if cold:
            clear_pipeline()

        session_id = f"bench-{label}-{context_len}-{path}-{rep}-{int(time.time())}"

        telemetry.enable(
            label=label,
            model=model_name,
            target_context=context_len,
            path=path,
            repeat=rep,
        )

        query = UserQuery(
            prompt=prompt,
            model_name=model_name,
            session_id=session_id,
            tokens_to_generate=tokens,
            dtype=torch.float16,
        )

        try:
            t0 = time.perf_counter()
            send_query(query, local, sm)
            wall = time.perf_counter() - t0

            if topology is None:
                topology = describe_pipeline(model_name)
            print("·", end="", flush=True)
        except Exception as e:
            print(f"\n    run {rep} failed: {type(e).__name__}: {e}")
        finally:
            telemetry.disable()

        sm.delete_session(session_id)

    return topology


def summarize(csv_path, out_path):
    """Median per-condition statistics, the numbers that go in the paper."""
    if not os.path.exists(csv_path):
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    groups = {}
    for r in rows:
        key = (r["label"], r["target_context"], r["path"], r["phase"])
        groups.setdefault(key, []).append(r)

    def med(vals):
        return round(statistics.median(vals), 3) if vals else 0.0

    out = []
    for (label, ctx, path, phase), rs in sorted(groups.items()):
        compute = [float(r["master_compute_ms"]) for r in rs]
        rt = [float(r["roundtrip_ms"]) for r in rs]
        step = [float(r["step_ms"]) for r in rs]
        out.append({
            "label": label,
            "context_tokens": ctx,
            "path": path,
            "phase": phase,
            "n_tokens": len(rs),
            "master_compute_ms_median": med(compute),
            "roundtrip_ms_median": med(rt),
            "step_ms_median": med(step),
            "step_ms_min": round(min(step), 3),
            "step_ms_max": round(max(step), 3),
            "roundtrip_share": round(med(rt) / med(step), 3) if med(step) else 0,
            "tokens_per_sec": round(1000.0 / med(step), 2) if med(step) else 0,
            "hidden_bytes": rs[0].get("hidden_bytes", ""),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\n[Bench] Summary written to {out_path}\n")
    print(f"  {'context':>8} {'path':>5} {'phase':>8} {'compute':>9} "
          f"{'roundtrip':>10} {'step':>8} {'tok/s':>7}")
    print("  " + "-" * 62)
    for r in out:
        print(f"  {r['context_tokens']:>8} {r['path']:>5} {r['phase']:>8} "
              f"{r['master_compute_ms_median']:>8.1f}m "
              f"{r['roundtrip_ms_median']:>9.1f}m "
              f"{r['step_ms_median']:>7.1f}m {r['tokens_per_sec']:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="name for this hardware configuration, e.g. 2-node")
    ap.add_argument("--model", default="llama-3b")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--contexts", default="64,256,1024,2048")
    ap.add_argument("--quick", action="store_true",
                    help="2 repeats, 2 context lengths")
    args = ap.parse_args()

    if args.quick:
        args.repeats = 2
        args.contexts = "64,512"

    contexts = [int(c) for c in args.contexts.split(",")]

    local = LocalConfig.load()
    sm = SessionManager()
    tokenizer = get_tokenizer(local, args.model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{args.label}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"[Bench] configuration '{args.label}'")
    print(f"  machine  : {local.tailscale_ip} ({local.device})")
    print(f"  model    : {args.model}")
    print(f"  contexts : {contexts}")
    print(f"  tokens   : {args.tokens} per query")
    print(f"  repeats  : {args.repeats}")

    telemetry.clear()
    topology = None

    # Warm the pipeline once so the first measured run isn't paying for
    # discovery and model loading unless it is meant to.
    print("\n[Bench] warming up...")
    warm = UserQuery(prompt="hello", model_name=args.model,
                     session_id="bench-warmup", tokens_to_generate=4,
                     dtype=torch.float16)
    try:
        send_query(warm, local, sm)
        sm.delete_session("bench-warmup")
    except Exception as e:
        print(f"[Bench] warm-up failed: {e}")
        print("        Check the daemons are running and benchmarks exist.")
        return 1
    telemetry.clear()

    print("\n[Bench] Experiment 1/2: context length sweep (warm path)")
    for ctx in contexts:
        t = run_condition(local, sm, args.model, ctx, args.tokens,
                          args.repeats, args.label, cold=False,
                          tokenizer=tokenizer)
        topology = topology or t

    print("\n\n[Bench] Experiment 2/2: cold vs warm path")
    ctx = contexts[len(contexts) // 2]
    run_condition(local, sm, args.model, ctx, args.tokens,
                  max(2, args.repeats // 2), args.label, cold=True,
                  tokenizer=tokenizer)

    n = telemetry.write_csv(csv_path)
    if n:
        summarize(csv_path, os.path.join(RESULTS_DIR, f"{args.label}_summary.csv"))

    if topology:
        meta = os.path.join(RESULTS_DIR, f"{args.label}_topology.txt")
        with open(meta, "w", encoding="utf-8") as f:
            f.write(f"label: {args.label}\n")
            f.write(f"stages: {topology['stages']}\n")
            f.write(f"topology: {topology['topology']}\n")
            f.write(f"model: {args.model}\n")
            f.write(f"device: {local.device}\n")
        print(f"[Bench] Topology recorded: {topology['stages']} stage(s) — "
              f"{topology['topology']}")

    print(f"\n[Bench] Done. Next: run this on your other hardware "
          f"configurations, then\n        python plot_results.py results/*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())