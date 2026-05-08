import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
import torch
import time
import psutil
import os
from safetensors.torch import load_file

def setup_model(stopping_layer:int):
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "./llama-3b"
    starting_layer = stopping_layer + 1

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    config.num_hidden_layers = stopping_layer

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    state_a = {}

    state_a.update(load_file(f"./layers/embed_tokens.safetensors", device=device))
    for i in range(stopping_layer):
        state_a.update(load_file(f"./layers/layer_{i}.safetensors", device=device))
        print(f"Loaded layer {i}")

    
    model.load_state_dict(
        state_a,
        strict=False,
        assign=True
    )

    print(f"Load time: {time.time() - start:.2f}s")
    print("Machine A ready")
    # Verify real layers were loaded
    for i in range(stopping_layer):
        param = next(model.model.layers[i].parameters())
        print(f"Layer {i} device: {param.device}")  # should print 'cpu'

    # Verify meta layers were never loaded
    for i in range(stopping_layer, 28):
        param = next(model.model.layers[i].parameters())
        print(f"Layer {i} device: {param.device}")  # should print 'meta'

if __name__ == "__main__":
    setup_model(26)