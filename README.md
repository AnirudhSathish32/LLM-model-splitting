# Distributed LLM Inference

Run a large language model across multiple machines on your local network.
Each machine holds a slice of the model's layers — a model that doesn't fit
on any single device can run across all of them together. No cloud API, no
per-token costs, no data leaving your network.

Everything is built from scratch in PyTorch: the pipeline parallelism, the
TCP wire protocol, the query scheduler, and the per-layer weight splitting.
No external serving framework required.

## What it does

- **Splits a model across N machines** proportional to each machine's
  speed and available memory. A fast GPU gets more layers; a slow CPU gets
  fewer. The split is computed automatically from benchmarks — nothing to
  configure by hand.
- **Discovers machines automatically** over a Tailscale network. Start a
  daemon on a machine and it joins the next pipeline that gets built.
- **Reuses a live pipeline** across turns. The first query pays for model
  loading and connection setup; every later query skips all of it.
- **Interleaves concurrent queries.** A round-robin scheduler advances every
  in-flight request one token at a time, so two people querying at once both
  see their answers stream rather than one waiting for the other.
- **Serves multiple models at once**, unloading the least recently used one
  when memory gets tight.
- **Remembers conversations** across restarts, stored as JSON on the machine
  that started them.

Verified working on 2- and 3-machine pipelines with Llama 3.2 3B and Llama
3.1 8B, across mixed CUDA and CPU-only hardware. Output matches
single-machine inference to >0.999 cosine similarity, validated layer by
layer.

## How it works

```
   Machine A (GPU)          Machine B (GPU)          Machine C (CPU)
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │  layers 0..13  │      │  layers 14..21 │      │  layers 22..31 │
 │    "master"    │─────→│    "worker"    │─────→│     "tail"     │
 │                │ state│                │ state│   + lm_head    │
 └────────────────┘      └────────────────┘      └────────────────┘
        ↑ │                                              │
   prompt│ └────────────── token (return) ──────────────┘
                        Tailscale VPN
```

A prompt enters at the **master**, which runs its layers and sends the
resulting hidden state onward. Each **worker** in the middle receives a hidden
state, runs its own layers, and passes it along. The last machine — the
**tail** — runs the final layers plus the output head, producing one token,
and sends it straight back to the master.

That loop repeats once per token until the response is complete. With two
machines there is no worker, and the master and tail talk directly.

Each machine only ever holds its own layers. Weights are pre-split into
one file per layer, so a machine downloads and loads exactly what it needs.

**Why pipeline parallelism.** Every extra machine adds one network hop to
every token, so depth costs latency. What it buys is capacity: a model too
large for any single machine becomes runnable. Use the fewest machines that
fit the model.

## Requirements

- Python 3.10+
- [Tailscale](https://tailscale.com/) installed and logged in on every machine
- A [Hugging Face](https://huggingface.co/) account, to download model weights
- One machine with a GPU is recommended but not required

Machines can be a mix of Windows, Linux, and macOS, and a mix of GPU and
CPU-only. You need room for the model twice while splitting it (about 12 GB for a
3B model), but only the layer files afterwards.

## Install

Everything below runs on **every machine** that will take part.

### 1. Get the code

```bash
git clone <your-repo-url>
cd model_splitting
```

### 2. Create a virtual environment

This keeps the project's packages — particularly its specific PyTorch build —
away from your system Python.

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Your prompt should now be prefixed with `(.venv)`. **Activate it in every new
terminal** before running anything below — including `launch.py` later on.

If PowerShell refuses to run the activation script, allow local scripts once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Install dependencies

```bash
python install.py
```

This detects your GPU and NVIDIA driver, installs the matching PyTorch build
from the right wheel index, then installs everything in `requirements.txt`
(Transformers, safetensors, FastAPI, and the Hugging Face CLI among others).
It finishes by running a real CUDA kernel to confirm the build actually works
on your card.

PyTorch is deliberately **not** listed in `requirements.txt`. The correct
wheel depends on hardware pip cannot see, so a plain `pip install -r` would
give you a build that imports fine and then fails with
`no kernel image is available` the first time you run a layer.

If detection gets it wrong:

```bash
python install.py --dry-run     # show what it would install, change nothing
python install.py --cpu         # force the CPU-only build
python install.py --cuda 12.8   # force a specific CUDA version
```

Expected output ends with something like:

```
Verifying
  torch 2.7.0+cu128
  cuda_available True
  device NVIDIA GeForce RTX 5070 Ti
```

## Setup

### 1. Configure the machine

Create `config/local_config.json` (the system runs with defaults if this
file is absent, but you need to set at least `tailscale_ip`):

```json
{
  "device": "cuda",
  "tailscale_ip": "100.x.x.x",
  "model_path": "./models",
  "layers_path": "./layers",
  "overhead": 0.2,
  "debug": false
}
```

`tailscale_ip` is this machine's address — find it with `tailscale ip -4`.
Use `"cpu"` for `device` on machines without a GPU. `overhead` controls how much memory is held in reserve (0.2 means 20%
stays free for the KV cache, activations, and the OS). Raise it if you
hit out-of-memory errors; lower it to fit more layers on the machine.

You can edit all of these later from the Settings panel in the web UI.

### 2. Get the model files

The walkthrough below uses **Llama 3.2 3B Instruct**. Any Hugging Face
decoder-only model works the same way — Mistral, Qwen, Phi, and so on.

**Accept the license.** Llama models are gated. Visit
[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
while signed in and accept the terms. Approval is usually immediate. Skipping
this gives you a 403 on download.

**Log in.** Create a token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
(read access is enough), then:

```bash
hf auth login
```

Paste the token when prompted. If `hf` isn't recognized, use
`huggingface-cli login` instead. The login is saved to your home directory,
so you only do this once per machine.

**Download into a folder named for the model.** The folder name is what you
will refer to everywhere else, so keep it short:

```bash
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3b
```

If `hf` isn't recognized, use `huggingface-cli download` instead — same
command, older name.

About 6 GB. `models/llama-3b/` should now contain `config.json`,
`tokenizer.json`, and a couple of `model-0000X-of-0000X.safetensors` files.

**Split it into per-layer files:**

```bash
python create_layer_files.py llama-3b
```

This writes `layers/llama-3b/layer_0.safetensors` through `layer_27.safetensors`,
plus `embed_tokens`, `norm`, and `head`. Run it **once** on any machine, then
copy the `layers/` folder to the others — or copy only the layer files a given
machine will be assigned, if you know the split in advance.

**Then delete almost all of it.** Once split, the full weight files are never
read again at runtime. Only five small files must remain in
`models/llama-3b/`:

```
config.json                generation_config.json
tokenizer.json             tokenizer_config.json
special_tokens_map.json
```

Roughly 9 MB instead of 6 GB. Safe to delete: `model-0000X.safetensors`,
`model.safetensors.index.json`, `original/`, `.cache/`, `LICENSE.txt`,
`README.md`. At runtime the code only calls `AutoConfig` and `AutoTokenizer`,
which read JSON and never touch the weights — the layer files supply
everything else.

This applies to every path, including single-machine mode — that rebuilds
the complete model from the layer files too, so there is no reason to keep
the originals.

**Repeat on every machine.** Each one needs the five JSON files and its own
layer files. You can copy them across rather than downloading again.

### 3. Benchmark each machine

On every machine, for every model you plan to run:

```bash
python benchmark.py llama-3b
```

This runs a short benchmark (about 30 seconds) measuring how fast one
layer executes and how much memory is free, writing the result to
`benchmark/`. The split algorithm reads
these — **a machine without a benchmark for a model is skipped when the
pipeline is built.**

Re-run it if you change hardware or move a machine.

## Launch

On every machine, with the virtual environment active:

```bash
python launch.py
```

This starts the daemon that makes the machine available as a pipeline stage,
serves the web UI, and opens a browser at `http://localhost:8000`.

| Flag | Effect |
|---|---|
| `--daemon-only` | Take part in inference with no UI |
| `--no-browser` | Start the UI but don't open a browser |
| `--port 8100` | Serve the UI on a different port |
| `--daemon-port N` | Change the daemon port (default 65433) |

You can send queries from whichever machine you're sitting at. The UI is
local to each machine — `localhost:8000` on Machine A only serves Machine
A's browser. Each machine keeps its own conversations.

There is also a terminal client:

```bash
python main.py                    # interactive
python main.py "What is 2+2?"     # one-shot
```

## Using it

The first query on a model runs the **cold path**: discover machines,
collect benchmarks, compute the split, load layers, connect the chain, then
generate. Expect several seconds.

Every query after that runs the **warm path**, reusing the loaded model, the
open connections, and the conversation cache. Expect it to start
immediately.

The pipeline strip above the chat shows which machine holds which layers, and
which one you're sitting at. The **Activity** panel on the right shows live
logs from every machine in the pipeline, each in its own collapsible section
— including the machines you aren't sitting at.

### Seeing concurrent queries interleave

Open the UI on two machines, start a conversation on each, and send both at
once. In the Activity panel on the master you'll see token lines alternating
between the two session IDs — the scheduler advancing both requests one step
at a time.

Two queries in the *same* conversation are handled in order, since a
follow-up question needs the previous answer to exist.

## Performance notes

These are properties of the approach, not bugs — worth knowing before you're
surprised by them.

**Prefill dominates on a slow tail machine.** The first query in a
conversation processes the entire prompt through every layer. On a GPU that
takes a second; on a CPU holding several layers, a 1500-token conversation
can take minutes. Later turns are fast because the cache retains all previous tokens — only
the newly added message gets processed through the layers.

**Switching models mid-conversation is expensive.** The cache is per
conversation *and* per model, and a cache built by one model can't be used by
another. Switching re-processes the whole history through the new model.
Starting a fresh conversation is much faster.

**Check the balance ratio.** The split table prints one — 1.0 is perfect,
and low numbers mean one machine is holding everyone else up. A CPU tail at
0.16 means the CPU is doing essentially all the waiting. Giving the slow machine fewer layers (by lowering `overhead` on the fast
machine so it claims more) improves the balance at the cost of more memory
pressure on the GPU.

**Models compete for memory.** Running two models at once needs room for
both. When memory is tight the least recently used model is unloaded, and
using it again pays the cold path. On a 16 GB GPU, Llama 3B and Llama 8B
genuinely cannot coexist.

## Layout

```
launch.py           Start the daemon and the web UI
install.py          Hardware-aware dependency installer
main.py             Terminal client

daemon.py           Per-machine service: accepts queries, owns peers
inference_peer.py   One machine's role in a pipeline; the wire protocol
model.py            Loads a layer range; role-specific forward passes
scheduler.py        Round-robin interleaving of concurrent requests
user_query.py       Orchestration: discovery, warm/cold paths
hardware.py         Computes the layer split from benchmarks
session.py          Conversations, persisted as JSON
serialization.py    Tensor wire format
protocol.py         Length-prefixed message framing
logbuffer.py        Captures console output for the Activity panel
web/                FastAPI server and the UI

benchmark.py            Measure this machine's speed for a model
create_layer_files.py   Split a model into per-layer files
```

## Troubleshooting

**A machine shows as unavailable.** It has no benchmark for that model. Run
`python benchmark.py <model>` on it.

**`403 Forbidden` downloading a model.** You haven't accepted the license on
the model's Hugging Face page, or you aren't logged in. Accept the terms while
signed in, then `hf auth login`.

**`hf: command not found`.** The virtual environment isn't active, or
`install.py` hasn't run yet. Activate it and re-run `python install.py`. On
`huggingface_hub` older than 0.34 the command is `huggingface-cli` instead.

**`ModuleNotFoundError` on a package you installed.** Almost always a
deactivated virtual environment in a new terminal. Check for `(.venv)` in
your prompt.

**"missing N of M layer files".** The split is incomplete for that model.
Re-run `python create_layer_files.py <model>` — that step needs the original
weights, so re-download them first if you have already deleted them.

**`no kernel image is available`.** Wrong PyTorch build for your GPU. Re-run
`python install.py`, or force a version with `--cuda 12.8`.

**`PermissionError` or `WinError 10013` binding a port.** Windows reserves
port ranges for Hyper-V and WSL. Check with
`netsh interface ipv4 show excludedportrange protocol=tcp` and move the port
range in `_model_port()` in `user_query.py` if yours collides.

**Tailscale says "no internet access."** Usually a Windows connectivity check
failing, not a real problem — the machines only need to reach each other.
Confirm with `ping <other machine's tailscale ip>`.

**A machine went offline mid-conversation.** The next query rebuilds the
pipeline across whatever is still available, down to a single machine.

## Status

A working prototype — functional for testing, demos, and personal use on a
small network. Not hardened for production — no access control, no monitoring, no TLS on
the daemon protocol beyond what Tailscale provides.

### Known limitations

- Layer assignments only change between queries, not during one
- All machines must be reachable on the same tailnet
- Every machine must run the same version — an older daemon won't understand
  newer messages
- Traffic between machines is encrypted by Tailscale but the daemon
  protocol itself has no authentication — any device on your tailnet can
  send queries
- Mobile devices aren't supported yet
- Memory is either GPU or system RAM per machine, never both together

### Possible next steps

Hybrid GPU+RAM placement within a machine; hosting pre-split layers so a new
machine downloads only its own; per-node timing telemetry in the UI;
quantized weights so weaker machines can hold more layers; an authenticated
API for multi-user serving.
