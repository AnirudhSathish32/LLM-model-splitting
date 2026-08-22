"""
plot_results.py

Turns the benchmark CSVs into the figures for the writeup.

    python plot_results.py results/*.csv

Writes vector PDFs (what a paper wants) and PNGs (for slides) to
figures/. Each figure is skipped with an explanation if the data it
needs is missing, so partial results still produce partial output.

Figures:
    fig1_latency_breakdown  where per-token time goes: master compute
                            vs. everything downstream
    fig2_pipeline_depth     tokens/sec against number of stages
    fig3_prefill_decode     how the two phases scale with context length
    fig4_warm_vs_cold       cost of rebuilding a pipeline
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required:  pip install matplotlib")
    sys.exit(1)

OUT_DIR = "figures"

# Muted, print-safe, and distinguishable in greyscale.
INK = "#1B2733"
COMPUTE = "#4C72B0"
NETWORK = "#DD8452"
ACCENT = "#55A868"
GREY = "#8C8C8C"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def load(paths):
    """Read every per-token CSV, skipping summary and topology files."""
    rows = []
    for p in paths:
        if p.endswith("_summary.csv") or p.endswith("_topology.txt"):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in ("master_compute_ms", "roundtrip_ms", "step_ms"):
                    if k in r:
                        r[k] = float(r[k])
                for k in ("token_index", "input_tokens", "context_tokens",
                          "target_context", "hidden_bytes"):
                    if k in r and r[k] != "":
                        r[k] = int(float(r[k]))
                rows.append(r)
    return rows


def stage_counts(paths):
    """Read stage counts recorded alongside each run."""
    counts = {}
    for p in paths:
        base = p[:-4] if p.endswith(".csv") else p
        topo = base + "_topology.txt"
        if not os.path.exists(topo):
            continue
        label = stages = None
        for line in open(topo, encoding="utf-8"):
            if line.startswith("label:"):
                label = line.split(":", 1)[1].strip()
            if line.startswith("stages:"):
                stages = int(line.split(":", 1)[1].strip())
        if label and stages:
            counts[label] = stages
    return counts


def med(vals):
    return statistics.median(vals) if vals else 0.0


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.pdf (and .png)")


# ── Figure 1 ────────────────────────────────────────────────

def fig_latency_breakdown(rows):
    """Stacked bars: master compute vs. everything downstream, per config."""
    decode = [r for r in rows if r.get("phase") == "decode"
              and r.get("path") == "warm"]
    if not decode:
        print("  fig1 skipped: no warm decode rows")
        return

    by_label = defaultdict(list)
    for r in decode:
        by_label[r["label"]].append(r)

    labels = sorted(by_label)
    compute = [med([r["master_compute_ms"] for r in by_label[l]]) for l in labels]
    downstream = [med([r["roundtrip_ms"] for r in by_label[l]]) for l in labels]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = range(len(labels))

    ax.bar(x, compute, label="Master compute", color=COMPUTE, width=0.55)
    ax.bar(x, downstream, bottom=compute,
           label="Network + downstream stages", color=NETWORK, width=0.55)

    tallest = max(c + d for c, d in zip(compute, downstream)) or 1
    for i, (c, d) in enumerate(zip(compute, downstream)):
        total = c + d
        ax.text(i, total + tallest * 0.03, f"{total:.0f} ms",
                ha="center", va="bottom", fontsize=8)
        # Only annotate the share when there is a downstream stage and the
        # segment is tall enough to hold text.
        if d > tallest * 0.08:
            ax.text(i, c + d / 2, f"{d / total * 100:.0f}%",
                    ha="center", va="center", fontsize=8, color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title("Where per-token time goes (decode, warm path)")
    # Headroom for the value labels, and a legend below so it cannot
    # collide with the tallest bar.
    ax.set_ylim(0, tallest * 1.18)
    ax.legend(frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig1_latency_breakdown")


# ── Figure 2 ────────────────────────────────────────────────

def fig_pipeline_depth(rows, counts):
    """Throughput against pipeline depth."""
    decode = [r for r in rows if r.get("phase") == "decode"
              and r.get("path") == "warm"]
    if not decode or not counts:
        print("  fig2 skipped: needs *_topology.txt for two or more configs")
        return

    by_label = defaultdict(list)
    for r in decode:
        by_label[r["label"]].append(r["step_ms"])

    pts = sorted((counts[l], 1000.0 / med(v), l)
                 for l, v in by_label.items() if l in counts and med(v))
    if len(pts) < 2:
        print("  fig2 skipped: need at least two configurations")
        return

    stages = [p[0] for p in pts]
    tps = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(stages, tps, "o-", color=COMPUTE, linewidth=1.6, markersize=6)
    for s, t, lbl in pts:
        ax.annotate(f"{lbl}\n{t:.1f} tok/s", (s, t),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8)

    ax.set_xlabel("Pipeline stages (machines)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput against pipeline depth")
    ax.set_xticks(stages)
    ax.set_ylim(0, max(tps) * 1.35)
    save(fig, "fig2_pipeline_depth")


# ── Figure 3 ────────────────────────────────────────────────

def fig_prefill_decode(rows):
    """How each phase scales with context length."""
    warm = [r for r in rows if r.get("path") == "warm"]
    if not warm:
        print("  fig3 skipped: no warm rows")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    plotted = False

    for phase, colour, marker in (("prefill", NETWORK, "s"),
                                  ("decode", COMPUTE, "o")):
        pts = defaultdict(list)
        for r in warm:
            if r.get("phase") == phase:
                pts[r["target_context"]].append(r["step_ms"])
        if not pts:
            continue
        xs = sorted(pts)
        ys = [med(pts[x]) for x in xs]
        ax.plot(xs, ys, marker + "-", label=phase.capitalize(),
                color=colour, linewidth=1.6, markersize=5)
        plotted = True

    if not plotted:
        print("  fig3 skipped: no phase data")
        plt.close(fig)
        return

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Step latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_title("Prefill scales with context; decode does not")
    ax.legend(frameon=False)
    save(fig, "fig3_prefill_decode")


# ── Figure 4 ────────────────────────────────────────────────

def fig_warm_vs_cold(rows):
    """Cost of the first token when the pipeline must be rebuilt."""
    firsts = defaultdict(list)
    for r in rows:
        if r.get("token_index") == 0:
            firsts[(r["label"], r["path"])].append(r["step_ms"])

    labels = sorted({k[0] for k in firsts})
    have_both = [l for l in labels
                 if (l, "warm") in firsts and (l, "cold") in firsts]
    if not have_both:
        print("  fig4 skipped: need both warm and cold runs")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = range(len(have_both))
    w = 0.35

    warm = [med(firsts[(l, "warm")]) for l in have_both]
    cold = [med(firsts[(l, "cold")]) for l in have_both]

    ax.bar([i - w / 2 for i in x], warm, w, label="Warm path", color=COMPUTE)
    ax.bar([i + w / 2 for i in x], cold, w, label="Cold path", color=NETWORK)

    for i, (wv, cv) in enumerate(zip(warm, cold)):
        if wv:
            ax.text(i + w / 2, cv * 1.02, f"{cv / wv:.1f}x",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(have_both)
    ax.set_ylabel("First-token latency (ms)")
    ax.set_title("Cost of rebuilding the pipeline")
    ax.legend(frameon=False)
    save(fig, "fig4_warm_vs_cold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", default=["results/*.csv"])
    args = ap.parse_args()

    paths = []
    for pattern in args.csvs:
        paths.extend(glob.glob(pattern))
    paths = [p for p in paths if not p.endswith("_summary.csv")]

    if not paths:
        print("No result CSVs found. Run bench_latency.py first.")
        return 1

    print(f"Reading {len(paths)} file(s): {', '.join(os.path.basename(p) for p in paths)}")
    rows = load(paths)
    print(f"  {len(rows)} token records")
    if not rows:
        return 1

    counts = stage_counts(paths)
    setup_style()

    print("\nGenerating figures")
    fig_latency_breakdown(rows)
    fig_pipeline_depth(rows, counts)
    fig_prefill_decode(rows)
    fig_warm_vs_cold(rows)

    print(f"\nFigures in {OUT_DIR}/ — use the PDFs in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())