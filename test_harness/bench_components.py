"""
bench_components.py

Component-level validation, separate from the end-to-end latency runs.

Two independent experiments, each with its own dataset and metric, so
their numbers are not comparable to each other or to the end-to-end
pass rates. Report them in their own table.

  1. Cost model accuracy
     The allocator assumes a stage costs assigned_layers x layer_time_s.
     Only the master's stage time is directly observable, so this checks
     the model there: predicted vs. measured, across whatever layer
     counts the available configurations produce.

  2. Scheduler fairness
     Fires K concurrent requests and measures how evenly the scheduler
     distributes tokens between them, and whether any request is starved.

Usage:
    python bench_components.py --label 2-node                 # both
    python bench_components.py --label 2-node --only fairness
    python bench_components.py --label 2-node --concurrent 3

Writes results/components_<label>.json and prints the tables.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import telemetry
from config import LocalConfig
from session import SessionManager
from user_query import UserQuery, send_query, get_pipeline_info

RESULTS_DIR = "results"

_FILLER = (
    "The network carries the hidden state between machines in the pipeline "
    "and each stage adds latency to every token that is produced. "
)


def make_prompt(target_tokens):
    words = _FILLER.split()
    return " ".join((words * ((target_tokens // len(words)) + 2))[:target_tokens])


def load_benchmark(local, model_name):
    """This machine's own benchmark — the allocator's input."""
    path = os.path.join("./benchmark", f"{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# 1. Cost model accuracy
# ══════════════════════════════════════════════════════════════

def experiment_cost_model(local, sm, model_name, repeats, tokens):
    """
    Compare the allocator's predicted master stage time against measured.

    predicted_ms = assigned_layers x layer_time_s x 1000
    measured_ms  = median master_compute_ms over decode steps

    Decode steps only: prefill processes many tokens at once and is a
    different operating point than the per-token cost the allocator models.
    """
    print("\n" + "=" * 68)
    print("Experiment 1: cost model accuracy")
    print("=" * 68)

    bench = load_benchmark(local, model_name)
    if bench is None:
        print(f"  No benchmark for '{model_name}'. Run benchmark.py first.")
        return None

    layer_time_s = bench["layer_time_s"]
    print(f"  Benchmarked layer time : {layer_time_s * 1000:.4f} ms")

    telemetry.clear()
    telemetry.enable(experiment="cost_model", model=model_name)

    for rep in range(repeats):
        sid = f"costmodel-{rep}-{int(time.time())}"
        q = UserQuery(prompt=make_prompt(64), model_name=model_name,
                      session_id=sid, tokens_to_generate=tokens,
                      dtype=torch.float16)
        try:
            send_query(q, local, sm)
            print("  ·", end="", flush=True)
        except Exception as e:
            print(f"\n  run {rep} failed: {e}")
        finally:
            sm.delete_session(sid)

    telemetry.disable()
    rows = [r for r in telemetry.rows() if r.get("phase") == "decode"]
    if not rows:
        print("\n  No decode steps recorded.")
        return None

    assigned = rows[0]["master_layers"]
    measured = statistics.median(r["master_compute_ms"] for r in rows)
    predicted = assigned * layer_time_s * 1000.0
    error_pct = abs(predicted - measured) / measured * 100.0 if measured else 0.0

    # Per-layer cost implied by the measurement, for comparison with the
    # single-layer benchmark. Divergence here means the linear assumption
    # is breaking down.
    implied_per_layer = measured / assigned if assigned else 0.0

    print(f"\n  Master layers assigned : {assigned}")
    print(f"  Predicted stage time   : {predicted:.2f} ms")
    print(f"  Measured stage time    : {measured:.2f} ms "
          f"(median of {len(rows)} decode steps)")
    print(f"  Absolute error         : {error_pct:.1f}%")
    print(f"  Implied ms/layer       : {implied_per_layer:.4f} "
          f"(benchmarked {layer_time_s * 1000:.4f})")

    return {
        "layer_time_ms_benchmarked": round(layer_time_s * 1000, 4),
        "master_layers": assigned,
        "predicted_stage_ms": round(predicted, 3),
        "measured_stage_ms": round(measured, 3),
        "error_pct": round(error_pct, 2),
        "implied_ms_per_layer": round(implied_per_layer, 4),
        "decode_steps": len(rows),
    }


# ══════════════════════════════════════════════════════════════
# 2. Scheduler fairness
# ══════════════════════════════════════════════════════════════

def jains_index(values):
    """
    Jain's fairness index: 1.0 when every request gets an equal share,
    1/n in the worst case where one request takes everything.
    """
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    return (sum(values) ** 2) / (n * sum(v * v for v in values))


def experiment_fairness(local, sm, model_name, n_concurrent, tokens):
    """
    Fire N concurrent requests on distinct sessions and measure how the
    scheduler interleaves them.

    Distinct sessions matter: same-session queries serialize by design,
    since a follow-up needs the previous answer.
    """
    print("\n" + "=" * 68)
    print(f"Experiment 2: scheduler fairness ({n_concurrent} concurrent requests)")
    print("=" * 68)

    telemetry.clear()
    telemetry.enable(experiment="fairness", model=model_name,
                     concurrency=n_concurrent)

    errors = []

    def fire(i):
        sid = f"fair-{i}-{int(time.time())}"
        q = UserQuery(prompt=make_prompt(48), model_name=model_name,
                      session_id=sid, tokens_to_generate=tokens,
                      dtype=torch.float16)
        try:
            send_query(q, local, sm)
        except Exception as e:
            errors.append(f"request {i}: {type(e).__name__}: {e}")
        finally:
            try:
                sm.delete_session(sid)
            except Exception:
                pass

    threads = [threading.Thread(target=fire, args=(i,))
               for i in range(n_concurrent)]

    print(f"  Launching {n_concurrent} requests...")
    t0 = time.perf_counter()
    for t in threads:
        t.start()
        time.sleep(0.05)      # slight stagger so they do not collide on setup
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    telemetry.disable()

    if errors:
        for e in errors:
            print(f"  {e}")

    rows = sorted(telemetry.rows(), key=lambda r: r["wall_time"])
    if not rows:
        print("  No tokens recorded.")
        return None

    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)

    sessions = list(by_session)
    counts = [len(by_session[s]) for s in sessions]

    # Interleaving: what fraction of consecutive tokens switched request?
    # 0 means fully sequential, near 1 means alternating every token.
    switches = sum(1 for a, b in zip(rows, rows[1:])
                   if a["session"] != b["session"])
    switch_rate = switches / max(1, len(rows) - 1)

    # Starvation: the longest a request waited between its own tokens,
    # relative to the median gap. A request left behind shows a large ratio.
    worst_gap_ratio = 0.0
    gap_detail = {}
    for s, rs in by_session.items():
        times = [r["wall_time"] for r in rs]
        gaps = [b - a for a, b in zip(times, times[1:])]
        if len(gaps) < 2:
            continue
        med = statistics.median(gaps)
        ratio = (max(gaps) / med) if med > 0 else 0.0
        gap_detail[s] = {
            "median_gap_ms": round(med * 1000, 2),
            "max_gap_ms": round(max(gaps) * 1000, 2),
            "ratio": round(ratio, 2),
        }
        worst_gap_ratio = max(worst_gap_ratio, ratio)

    # Did every request start promptly, or did later ones wait for
    # earlier ones to finish? This is what the starvation bug looked like.
    first_token = {s: min(r["wall_time"] for r in by_session[s])
                   for s in sessions}
    start_spread = (max(first_token.values()) - min(first_token.values())) * 1000

    fairness = jains_index(counts)

    print(f"\n  Requests completed     : {len(sessions)}/{n_concurrent}")
    print(f"  Tokens per request     : {counts}")
    print(f"  Jain's fairness index  : {fairness:.4f}  (1.0 = perfectly equal)")
    print(f"  Token switch rate      : {switch_rate:.3f}  "
          f"(1.0 = alternating every token)")
    print(f"  First-token spread     : {start_spread:.0f} ms  "
          f"(how long the last request waited to start)")
    print(f"  Worst gap ratio        : {worst_gap_ratio:.2f}x median  "
          f"(a starved request shows a large value)")
    print(f"  Wall time              : {wall:.2f} s")

    expected_switch = 1.0 - 1.0 / n_concurrent if n_concurrent > 1 else 0.0
    print(f"\n  Expected switch rate for round-robin: {expected_switch:.3f}")
    if n_concurrent > 1:
        verdict = ("interleaving as designed" if switch_rate > expected_switch * 0.7
                   else "NOT interleaving — requests ran sequentially")
        print(f"  Verdict: {verdict}")

    return {
        "concurrency": n_concurrent,
        "requests_completed": len(sessions),
        "tokens_per_request": counts,
        "jains_fairness_index": round(fairness, 4),
        "token_switch_rate": round(switch_rate, 4),
        "expected_switch_rate": round(expected_switch, 4),
        "first_token_spread_ms": round(start_spread, 1),
        "worst_gap_ratio": round(worst_gap_ratio, 2),
        "wall_time_s": round(wall, 2),
        "per_session_gaps": gap_detail,
        "errors": errors,
    }


# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="hardware configuration name, e.g. 2-node")
    ap.add_argument("--model", default="llama-3b")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrent", type=int, default=2,
                    help="concurrent requests for the fairness test")
    ap.add_argument("--only", choices=["cost", "fairness"],
                    help="run only one experiment")
    args = ap.parse_args()

    local = LocalConfig.load()
    sm = SessionManager()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[Components] configuration '{args.label}'")
    print(f"  machine : {local.tailscale_ip} ({local.device})")
    print(f"  model   : {args.model}")

    # Warm the pipeline so setup cost is not attributed to an experiment.
    print("\n  warming up...")
    warm = UserQuery(prompt="hello", model_name=args.model,
                     session_id="components-warmup", tokens_to_generate=4,
                     dtype=torch.float16)
    try:
        send_query(warm, local, sm)
        sm.delete_session("components-warmup")
    except Exception as e:
        print(f"  warm-up failed: {e}")
        print("  Check the daemons are running and benchmarks exist.")
        return 1

    info = get_pipeline_info(args.model)
    stages = len(info["stages"]) if info and info.get("stages") else 1
    print(f"  pipeline: {stages} stage(s)")

    out = {
        "label": args.label,
        "model": args.model,
        "stages": stages,
        "device": local.device,
        "timestamp": time.time(),
    }

    if args.only != "fairness":
        out["cost_model"] = experiment_cost_model(
            local, sm, args.model, args.repeats, args.tokens)

    if args.only != "cost":
        out["fairness"] = experiment_fairness(
            local, sm, args.model, args.concurrent, args.tokens)

    path = os.path.join(RESULTS_DIR, f"components_{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Components] written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())