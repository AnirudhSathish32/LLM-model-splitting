# Distributed LLM Inference System

## Overview

This project implements distributed transformer inference across multiple heterogeneous devices.

Instead of executing all transformer layers on a single machine, the model is partitioned across:

* Machine A
* Machine B

Machine A executes the first portion of the model, serializes intermediate hidden states, and transfers them over TCP. Machine B resumes execution from the split point and generates the next token.

The system supports:

* Layer-level model splitting
* Cross-device hidden state transfer
* KV-cache based autoregressive decoding
* Layer validation against monolithic inference
* Resource monitoring
* CPU/GPU heterogeneous execution
* Distributed inference without cloud infrastructure

---

# Project Structure

```text
.
├── config.py
├── layer_files.py
├── machine_a.py
├── machine_b.py
├── validation_a.py
├── validation_b.py
├── generation.py
├── layers/
├── handoff/
├── received/
└── llama-3b/
```

---

# Requirements

Install dependencies inside a virtual environment.

## Create virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# Install dependencies

```bash
pip install torch transformers accelerate safetensors psutil
```

---

# Model Setup

Download a HuggingFace-compatible LLaMA model locally.

Example:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./llama-3b
```

---

# IMPORTANT: Config Must Match Across Both Machines

Both Machine A and Machine B MUST use identical configuration values.

The following values must match exactly:

```python
model_path
stopping_layer
prompt
tokens_to_generate
dtype
```

If configurations differ between machines:

* hidden state dimensions may mismatch
* positional embeddings may mismatch
* KV-cache alignment may break
* tensor validation will fail
* generation outputs may diverge

The split architecture assumes both machines are operating on the exact same transformer configuration.

---

# Step 1 — Generate Layer Files

Before running distributed inference, layer checkpoints must be extracted from the original downloaded model.

Run:

```bash
python layer_files.py
```

This creates serialized layer checkpoint files inside:

```text
./layers/<model_name>/
```

Example:

```text
layers/
└── llama-3b/
    ├── layer_0.safetensors
    ├── layer_1.safetensors
    ├── ...
    ├── norm.safetensors
    └── head.safetensors
```

These files are used so each machine only loads its assigned transformer layers instead of the full model.

---

# Distributed Execution Flow

## Machine A

Machine A:

1. Loads the first partition of layers
2. Executes Split 1
3. Captures hidden states
4. Sends tensors over TCP
5. Receives next-token predictions from Machine B
6. Continues autoregressive decoding

---

## Machine B

Machine B:

1. Loads the second partition of layers
2. Receives hidden states from Machine A
3. Executes Split 2
4. Produces logits
5. Selects next token
6. Sends token back to Machine A

---

# Networking

The system currently uses:

* TCP sockets
* Tailscale networking

Machine A acts as the TCP server.

Machine B connects as the client.

Intermediate transformer tensors are transferred as serialized binary `.pt` files.

---

# Validation System

The validation pipeline compares:

* Monolithic inference
* Distributed split inference

Metrics collected include:

* Layer-wise cosine similarity
* Max tensor difference
* Mean tensor difference
* Resource usage
* Timing statistics

---

# Running the System

## IMPORTANT

Run the correct validation script depending on which machine you are using.

---

# Machine A

On Machine A run:

```bash
python validation_a.py
```

Machine A:

* hosts the TCP server
* performs Split 1
* sends hidden states
* receives generated tokens

---

# Machine B

On Machine B run:

```bash
python validation_b.py
```

Machine B:

* connects to Machine A
* performs Split 2
* generates next-token logits
* sends tokens back

---

# Current Results

Example benchmark results observed during testing:

## CPU Machine

* Full inference: ~327s
* Split inference: ~201s
* ~38% latency reduction

---

## GPU Machine

* VRAM reduction: ~67%
* Split execution slower due to per-token network round trips

---

# Technical Features

## Layer Splitting

The model is partitioned at a configurable transformer layer boundary.

Example:

```python
stopping_layer = 14
```

Machine A:

```text
Layers 0–13
```

Machine B:

```text
Layers 14–27
```

---

## KV Cache Support

The system maintains autoregressive KV caches across distributed execution.

This significantly reduces repeated transformer computation during generation.

---

## Layer Validation Hooks

Forward hooks capture intermediate hidden states from each transformer layer.

These activations are compared against monolithic generation outputs to verify correctness.

---

## Resource Monitoring

The project tracks:

* CPU utilization
* RAM usage
* GPU VRAM usage
* Inference latency
* Layer execution timing

---

# Future Work

Potential future improvements include:

* Dynamic layer splitting
* Adaptive runtime scheduling
* Multi-device inference clusters
* Token batching
* Streaming hidden-state transport
* Speculative decoding
* gRPC transport layer
* Distributed edge inference orchestration

---

# Research Direction

This project explores:

* distributed transformer inference
* heterogeneous compute scheduling
* cooperative edge AI
* low-resource LLM deployment
* runtime transformer partitioning

The goal is enabling larger language models to run across commodity consumer hardware without centralized cloud infrastructure.

---

# Notes

* Both machines must use the same tokenizer and model files.
* Layer counts must remain consistent across machines.
* Split boundaries must align between Machine A and Machine B.
* Tailscale IPs must be configured correctly before execution.
* Validation scripts assume identical prompts and generation settings.

---

# Example Workflow

## 1. Download model

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./llama-3b
```

## 2. Generate layer files

```bash
python layer_files.py
```

## 3. Configure both machines

Update:

```python
config.py
```

Ensure both machines use identical settings. 

## 4. Start Machine A

```bash
python validation_a.py
```

## 5. Start Machine B

```bash
python validation_b.py
```

---

# License

This project is intended for research and educational purposes.

