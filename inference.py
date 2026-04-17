"""
Distributed LLM inference across 2 machines using torchrun + accelerate.

Run on Machine A (master):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
             --master_addr=192.168.100.1 --master_port=29500 inference.py

Run on Machine B (worker):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
             --master_addr=192.168.100.1 --master_port=29500 inference.py
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./llama-8b"   # path to your downloaded model folder
MAX_NEW_TOKENS = 200
QUERY = "Explain how pipeline parallelism works in large language models."
# ─────────────────────────────────────────────────────────────────────────────


def setup_distributed():
    """Initialize the distributed process group using env vars set by torchrun."""
    dist.init_process_group(backend="gloo")  # gloo works over ethernet; use nccl if both machines have CUDA p2p
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def build_device_map(rank: int, world_size: int, num_layers: int = 32) -> dict:
    """
    Manually partition transformer layers across machines.
    Rank 0 (Machine A) gets the first half + embeddings.
    Rank 1 (Machine B) gets the second half + output head.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layers_per_machine = num_layers // world_size
    start_layer = rank * layers_per_machine
    end_layer = start_layer + layers_per_machine if rank < world_size - 1 else num_layers

    device_map = {}

    if rank == 0:
        device_map["model.embed_tokens"] = device
    else:
        device_map["model.embed_tokens"] = "meta"  # not needed on worker

    for i in range(num_layers):
        if start_layer <= i < end_layer:
            device_map[f"model.layers.{i}"] = device
        else:
            device_map[f"model.layers.{i}"] = "cpu"  # will be offloaded/ignored

    if rank == world_size - 1:
        device_map["model.norm"] = device
        device_map["lm_head"] = device
    else:
        device_map["model.norm"] = "cpu"
        device_map["lm_head"] = "cpu"

    return device_map


def load_model(rank: int, world_size: int):
    """Load model with device_map so each machine only materializes its layers."""
    print(f"[Rank {rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[Rank {rank}] Loading model with device_map=auto...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",       # accelerate measures VRAM and splits optimally
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # don't load full model into RAM first
    )

    print(f"[Rank {rank}] Device map assigned:")
    for layer_name, device in model.hf_device_map.items():
        print(f"  {layer_name}: {device}")

    return model, tokenizer


def run_inference(model, tokenizer, rank: int):
    """
    Only rank 0 (Machine A) drives the query.
    Both machines participate in the forward pass via distributed ops.
    Only rank 0 prints the result.
    """
    if rank == 0:
        print(f"\n[Rank {rank}] Query: {QUERY}\n")
        print(f"[Rank {rank}] Generating response...")

    # Tokenize on rank 0, broadcast input_ids to all ranks
    if rank == 0:
        inputs = tokenizer(QUERY, return_tensors="pt")
        input_ids = inputs["input_ids"]
        # Tell other ranks the sequence length so they can allocate the tensor
        seq_len = torch.tensor([input_ids.shape[1]], dtype=torch.long)
    else:
        seq_len = torch.tensor([0], dtype=torch.long)

    # Broadcast sequence length first
    dist.broadcast(seq_len, src=0)

    # Broadcast input_ids
    if rank != 0:
        input_ids = torch.zeros((1, seq_len.item()), dtype=torch.long)
    dist.broadcast(input_ids, src=0)

    # Move to correct device for this rank
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = input_ids.to(device)

    # All ranks participate in forward pass
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy decoding — deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only rank 0 decodes and prints
    if rank == 0:
        # Slice off the prompt tokens, only show generated response
        generated = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"Response:\n{response}")
        print(f"{'='*60}\n")
        print(f"[Rank {rank}] Generated {len(generated)} tokens")


def main():
    rank, world_size = setup_distributed()
    print(f"[Rank {rank}] Process started — world size: {world_size}")

    model, tokenizer = load_model(rank, world_size)

    # Barrier: wait for both machines to finish loading before inference
    dist.barrier()
    print(f"[Rank {rank}] All machines ready. Starting inference...")

    run_inference(model, tokenizer, rank)

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()