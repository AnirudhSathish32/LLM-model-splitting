# Distributed LLM Inference

**A distributed inference system that partitions a transformer across multiple
heterogeneous machines, assigning layers dynamically based on each machine's
measured compute speed and available memory.**

Machines communicate over a custom length-prefixed TCP protocol. A scheduler
interleaves concurrent requests so multiple users share one pipeline. Every
distributed forward pass is validated layer by layer against single-machine
inference.

Built from scratch in PyTorch — the pipeline parallelism, wire protocol,
query scheduler, and per-layer weight splitting are all implemented here
rather than delegated to a serving framework. A model that fits on none of
the machines individually runs across all of them together, with no cloud API
and no data leaving the local network.

---

## Why this project

Language models increasingly exceed the memory of the hardware people
actually own. The usual answers are to buy a larger GPU, stand up a
centralized inference server, or send requests to a cloud API.

This explores a fourth option: **distribute the model itself.** Rather than
requiring every machine to hold the complete model, each machine loads only
the layers assigned to it — and the assignment is computed from measured
hardware performance rather than configured by hand.

```
                    Distributed Pipeline

 ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
 │    Machine A    │    │    Machine B    │    │    Machine C    │
 │                 │    │                 │    │                 │
 │   Layers 0–13   │───►│  Layers 14–21   │───►│  Layers 22–31   │
 │       GPU       │    │       GPU       │    │       CPU       │
 │                 │    │                 │    │    + LM head    │
 └─────────────────┘    └─────────────────┘    └─────────────────┘
          │                                             │
          └───────────── generated token ◄──────────────┘

                       Tailscale + TCP
```

## Results

| Capability | Result |
|---|---|
| Correctness | >0.999 cosine similarity vs. single-machine inference, validated per layer |
| RAM reduction | Up to 92% on a memory-constrained node |
| VRAM reduction | Up to 67% |
| Layer assignment | Computed automatically from per-machine benchmarks |
| Pipelines verified | up to 5 nodes mixed GPU and CPU-only |
| Models tested | Llama 3.2 3B (28 layers), Llama 3.1 8B (32 layers) |
| Network protocol | Custom length-prefixed TCP, 21 message types |
| Concurrency | Multiple interleaved generations on one pipeline |
| Multi-model | Several models served at once, LRU eviction under memory pressure |
| Conversation state | Persisted across process restarts |

**Stack.** PyTorch · Transformers · safetensors · FastAPI · Tailscale

---

## How it works

### Inference flow

Each machine holds a contiguous range of transformer layers. A prompt enters
at the first machine and the hidden state travels the chain; the last machine
runs the LM head and returns a single token to the first, which appends it
and starts the next pass.

```
Prompt
  │
  ▼
Machine A ─── layers 0–13
  │
  │ hidden state
  ▼
Machine B ─── layers 14–21
  │
  │ hidden state
  ▼
Machine C ─── layers 22–31 + LM head
  │
  │ next token
  ▼
Machine A ─── repeat until complete
```

Each machine keeps a KV cache for its own layers, so previously processed
tokens are never recomputed. With two machines there is no middle stage and
the first and last talk directly.

### Components

```
                  ┌─────────────────────┐
                  │    Query client     │
                  │    Web UI / CLI     │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   Query scheduler   │
                  │   interleaves       │
                  │   concurrent work   │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Pipeline builder   │
                  │                     │
                  │ • machine discovery │
                  │ • benchmark lookup  │
                  │ • layer allocation  │
                  └──────────┬──────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐
     │ Machine A │────►│ Machine B │────►│ Machine C │
     │  daemon   │     │  daemon   │     │  daemon   │
     └───────────┘     └───────────┘     └───────────┘
           │                 │                 │
           └──────────── Tailscale ────────────┘
```

### Dynamic layer allocation

Every machine benchmarks its own layer execution speed, available memory, and
device type. The pipeline builder uses those measurements to balance
execution time across stages rather than splitting layers evenly.

```
Fast GPU     →  more layers
Medium GPU   →  fewer layers
Slow CPU     →  fewest layers
```

The allocation is recomputed whenever the set of available machines changes,
so the same model runs across heterogeneous hardware without assuming
identical devices.

### Cold path and warm path

The first request pays the full setup cost. Every request after it reuses
what is already running.

```
FIRST REQUEST                      SUBSEQUENT REQUESTS
─────────────                      ───────────────────
machine discovery                  query
     ↓                                ↓
hardware benchmarking              existing pipeline
     ↓                                ↓
layer allocation                   inference
     ↓
model loading
     ↓
TCP connection setup
     ↓
inference
```

### Concurrent query scheduling

Generation is inherently sequential — token *N* depends on token *N−1* — and
the pipeline is a shared resource. Rather than letting one generation hold it
until finished, the scheduler advances every in-flight request one token per
rotation:

```
Request A → token
Request B → token
Request A → token
Request B → token
        ...
```

Each request carries its own KV cache and session state. Hidden states are
tagged with a session ID, so downstream machines switch caches per token
without knowing that interleaving is happening.

### Tensor transport

The project implements its own TCP protocol rather than depending on an
external distributed inference framework. Tensors are serialized to a
language-independent binary layout:

```
┌─────────┬─────────┬─────────┬──────────────┐
│  dtype  │  rank   │  shape  │ tensor bytes │
└─────────┴─────────┴─────────┴──────────────┘
```

Messages are length-prefixed so a receiver knows exactly how many bytes
belong to each one. The 21 message types cover query initialization, tensor
and token transfer, pipeline control, machine discovery, lifecycle events,
and failure signalling.

This replaced a `torch.save`-based transport, removing roughly 1 KB of pickle
framing per message and keeping the wire format independent of Python — which
matters for adding non-Python devices later.

### Model storage

The original checkpoint is converted once into individual layer files:

```
layers/llama-3b/
├── embed_tokens.safetensors
├── layer_0.safetensors
├── layer_1.safetensors
├── ...
├── layer_27.safetensors
├── norm.safetensors
└── head.safetensors
```

Each machine loads only the files for its assigned range, so the full
multi-gigabyte checkpoint is never needed at inference time — including on
the single-machine path, which reassembles the complete model from the same
layer files.

### Correctness validation

Distributed inference has a failure mode that ordinary testing misses: **the
pipeline can produce fluent, plausible text while its internal computation is
already wrong.** No exception, no warning.

Every layer's output is therefore compared against a single-machine reference
implementation, measuring maximum absolute difference, mean absolute
difference, and cosine similarity, with cosine similarity > 0.999 as the
acceptance criterion at each boundary.

This caught several bugs that all produced convincing output — KV-cache
indexing and causal-mask errors among them. Specifics in
[Design notes](#design-notes).

### Performance characteristics

```
        ↓ memory required per machine
        ↓ hardware requirements
        ↑ network communication
        ↑ pipeline complexity
```

Every additional machine adds a network hop to every generated token.
Distributing a model buys **capacity and hardware flexibility, not lower
latency.** For constrained hardware, though, it makes otherwise impossible
models runnable — and concurrent throughput does improve, since the scheduler
keeps stages busy across requests.

---

## Quick start

On **each** machine, with [Tailscale](https://tailscale.com/) installed and
logged in:

```bash
git clone <your-repo-url> && cd model_splitting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python install.py                                   # picks the right PyTorch build

hf auth login                                       # paste a HuggingFace token
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3b
python create_layer_files.py llama-3b               # split into per-layer files
python benchmark.py llama-3b                        # measure this machine

python launch.py                                    # opens the chat UI
```

Set `tailscale_ip` in `config/local_config.json` first — `tailscale ip -4`
prints it. Detailed walkthrough below.

## Requirements

- Python 3.10+
- [Tailscale](https://tailscale.com/) installed and logged in on every machine
- A [Hugging Face](https://huggingface.co/) account, for gated models
- A GPU on at least one machine is recommended, not required

Windows, Linux, and macOS all work and can be mixed in one pipeline. You need
room for the model twice while splitting it (about 12 GB for a 3B model), but
only the layer files afterwards.

## Install

Run on **every machine** that will take part.

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

Your prompt should now show `(.venv)`. **Activate it in every new terminal**
before running anything below, including `launch.py`.

If PowerShell blocks the activation script, allow local scripts once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python install.py
```

This detects your GPU and NVIDIA driver, installs the matching PyTorch build
from the correct wheel index, then installs everything in `requirements.txt`.
It finishes by running a real CUDA kernel to confirm the build works on your
card.

PyTorch is deliberately **not** in `requirements.txt`: the correct wheel
depends on hardware pip cannot detect, so a plain `pip install -r` gives you
a build that imports fine and then fails with `no kernel image is available`
the first time you run a layer.

If detection gets it wrong:

```bash
python install.py --dry-run     # show what it would install, change nothing
python install.py --cpu         # force the CPU-only build
python install.py --cuda 12.8   # force a specific CUDA version
```

## Setup

### 1. Configure the machine

Find this machine's Tailscale address:

```bash
tailscale ip -4
```

Create `config/local_config.json` (defaults apply if absent, but
`tailscale_ip` must be set):

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

Use `"cpu"` for `device` on machines without a GPU. `overhead` is how much
memory stays in reserve (0.2 means 20% free for the KV cache, activations,
and the OS) — raise it if you hit out-of-memory errors, lower it to fit more
layers.

All of these are also editable from the Settings panel in the web UI.

### 2. Get the model files

The walkthrough uses **Llama 3.2 3B Instruct**. Any HuggingFace decoder-only
model works the same way — Mistral, Qwen, Phi, and so on.

**Accept the license.** Llama models are gated. Visit
[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
while signed in and accept the terms; approval is usually immediate. Skipping
this gives a 403 on download.

**Log in.** Create a read token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
then:

```bash
hf auth login
```

If `hf` isn't recognized, use `huggingface-cli login` — same command, older
name. The login is saved to your home directory, so it's once per machine.

**Download into a folder named for the model.** That folder name is what you
refer to everywhere else, so keep it short:

```bash
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3b
```

About 6 GB.

**Split it into per-layer files:**

```bash
python create_layer_files.py llama-3b
```

This writes `layers/llama-3b/layer_0.safetensors` through
`layer_27.safetensors`, plus `embed_tokens`, `norm`, and `head`. Run it
**once** on any machine, then copy `layers/` to the others — or copy only the
files a given machine will be assigned.

**Then delete almost all of it.** Once split, the original weights are never
read again. Only five small files must remain in `models/llama-3b/`:

```
config.json                generation_config.json
tokenizer.json             tokenizer_config.json
special_tokens_map.json
```

Roughly 9 MB instead of 6 GB. Safe to delete: `model-0000X.safetensors`,
`model.safetensors.index.json`, `original/`, `.cache/`, `LICENSE.txt`,
`README.md`. At runtime the code only calls `AutoConfig` and `AutoTokenizer`,
which read JSON and never touch the weights — the layer files supply
everything else, on every path including single-machine mode.

**Repeat on every machine.** Each needs the five JSON files and its own layer
files; copying beats downloading again.

### 3. Benchmark each machine

On every machine, for every model you plan to run:

```bash
python benchmark.py llama-3b
```

A short benchmark (about 30 seconds) measuring layer execution speed and free
memory, written to `benchmark/`. The allocation algorithm reads these — **a
machine with no benchmark for a model is skipped when the pipeline is
built.** Re-run it if you change hardware.

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
| `--no-browser` | Start the UI without opening a browser |
| `--port 8100` | Serve the UI on a different port |
| `--daemon-port N` | Change the daemon port (default 65433) |

The UI is local to each machine — `localhost:8000` on Machine A serves only
Machine A's browser. Each machine keeps its own conversations.

There is also a terminal client:

```bash
python main.py                    # interactive
python main.py "What is 2+2?"     # one-shot
```

## Using it

The first query on a model runs the cold path and takes several seconds.
Every query after reuses the loaded model, open connections, and conversation
cache, and starts immediately.

The pipeline strip above the chat shows which machine holds which layers and
which one you're sitting at. The **Activity** panel shows live logs from
every machine in the pipeline, each collapsible — including machines you
aren't sitting at.

**To watch queries interleave:** open the UI on two machines, start a
conversation on each, and send both at once. The Activity panel on the first
machine shows token lines alternating between session IDs. Two queries in the
*same* conversation run in order, since a follow-up needs the previous
answer.

### Things that will surprise you

These are properties of the approach, not defects.

**Prefill dominates on a slow machine.** The first query in a conversation
runs the entire prompt through every layer. On a GPU that's a second; on a
CPU holding several layers, a 1500-token conversation can take minutes. Later
turns are fast because the cache retains previous tokens — only the new
message is processed.

**Switching models mid-conversation is expensive.** The cache is per
conversation *and* per model, and one model's cache can't be reused by
another. Switching reprocesses the whole history. Starting a fresh
conversation is much faster.

**Watch the balance ratio.** The split table prints one; 1.0 is perfect. A
CPU tail at 0.16 means the CPU is doing essentially all the waiting. Lowering
`overhead` on the fast machine lets it claim more layers, at the cost of
memory pressure there.

**Models compete for memory.** Two models at once need room for both. Under
pressure the least recently used is unloaded and pays the cold path next
time. On a 16 GB GPU, Llama 3B and Llama 8B genuinely cannot coexist.

## Troubleshooting

**A machine shows as unavailable.** No benchmark for that model. Run
`python benchmark.py <model>` on it.

**`403 Forbidden` downloading a model.** License not accepted, or not logged
in. Accept the terms while signed in, then `hf auth login`.

**`hf: command not found`.** Virtual environment not active, or `install.py`
hasn't run. On `huggingface_hub` older than 0.34 the command is
`huggingface-cli`.

**`ModuleNotFoundError` on a package you installed.** Almost always a
deactivated virtual environment in a new terminal. Check for `(.venv)`.

**"missing N of M layer files".** The split is incomplete. Re-run
`python create_layer_files.py <model>` — that needs the original weights, so
re-download them first if you've deleted them.

**`no kernel image is available`.** Wrong PyTorch build for your GPU. Re-run
`python install.py`, or force it with `--cuda 12.8`.

**`PermissionError` / `WinError 10013` binding a port.** Windows reserves
port ranges for Hyper-V and WSL. Check
`netsh interface ipv4 show excludedportrange protocol=tcp` and move the range
in `_model_port()` in `user_query.py` if yours collides.

**Tailscale says "no internet access."** Usually a Windows connectivity check
failing, not a real problem — machines only need to reach each other. Confirm
with `ping <other machine's tailscale ip>`.

**A machine went offline mid-conversation.** The next query rebuilds across
whatever remains, down to a single machine.

## Project structure

```
launch.py               Starts the daemon and web UI
install.py              Hardware-aware dependency installer
main.py                 Terminal client

daemon.py               Per-machine inference service
inference_peer.py       A machine's role in a pipeline; wire protocol
model.py                Layer loading and role-specific forward passes
scheduler.py            Concurrent request scheduling
user_query.py           Query orchestration and pipeline construction
hardware.py             Benchmarking and layer allocation
session.py              Conversation persistence
serialization.py        Tensor serialization
protocol.py             TCP message framing
logbuffer.py            Console capture for the Activity panel
web/                    FastAPI server and web interface

benchmark.py            Measure this machine's speed for a model
create_layer_files.py   Per-layer weight extraction
```

---

## Design notes

Detail on the problems that shaped the implementation, for anyone reading the
source.

### What layer-level validation caught

The >0.999 cosine similarity check exists because three separate bugs each
produced fluent, entirely convincing output:

- A **KV cache keyed by the wrong name**, so cached entries were written and
  then never found — correct output, silently no cache reuse.
- **Cache indices left pointing at pre-split positions** after the layer stack
  was sliced, so attention read from the wrong offsets.
- A **causal mask that stopped being applied** once the stack was rebuilt,
  letting positions attend forward in the sequence.

None raised an exception. Without a per-layer numerical oracle, all three
would have shipped.

### A scheduling heuristic that was exactly backwards

The first scheduler prioritized decode steps over prefill, reasoning that
finishing an in-progress request frees the pipeline sooner. It starved new
requests completely: a running request's decode always outranked a waiting
request's prefill, so a second query never started until the first finished.

That heuristic is correct for continuous batching, where decode steps across
requests are batched into one forward pass. It is actively wrong for
one-step-at-a-time scheduling, where it degenerates into strict FIFO.
Replaced with least-recently-stepped ordering.

### Protocol decisions

**Responses travel on the connection the query arrived on.** An earlier design
had the last machine open a separate socket back to the initiator, which
meant port allocation, bind conflicts on restart, and query IDs to match
responses to requests. Replying on the existing connection makes
request/response pairing implicit and scales to concurrent queries for free.

**Two machines share one full-duplex socket.** The original used a separate
port per direction; one was silently blocked by Windows Firewall, producing a
hang with no error. TCP is already bidirectional and the two ends alternate
rather than transmit simultaneously, so one connection carries both
directions with message types distinguishing them.

### Failure handling without a coordinator

There is no central controller, so each machine must independently notice
when the pipeline it belongs to has stopped existing.

A machine leaving used to leave behind a peer that *looked* healthy — model
loaded, sockets present — but whose generation loop had died. Later queries
matched the cached pipeline, routed onto dead sockets, and hung. Liveness is
now based on whether the generation thread or scheduler is actually running,
a departing machine announces itself before exiting, and a connection-level
failure releases the peer so the next query rebuilds.

Memory eviction had a related fault: it ran while holding the lock every
other query needed, so unloading one model stalled every other model for the
duration. Eviction now selects a victim under the lock and does the slow work
outside it, and never evicts a model that is mid-generation.

---

## Limitations

A working research prototype, intended for experimentation, demonstrations,
and research into distributed and edge inference rather than production
deployment.

- Layer assignments change between queries, not during an active generation
- All machines must be reachable on the same Tailscale network
- All machines must run compatible versions
- The daemon protocol has no authentication of its own beyond Tailscale's
  network-level access control
- Memory is managed as either GPU memory or system RAM per machine, never both
- Network communication is the dominant latency cost for fast GPU pipelines
- Mobile devices are not yet supported

## Future work

Dynamic rebalancing during active generation · hybrid GPU + CPU memory
placement · quantized distributed weights · token batching · speculative
decoding · per-node runtime telemetry · hosted layer files so a new machine
downloads only its own · authenticated multi-user API · mobile and edge
device support
