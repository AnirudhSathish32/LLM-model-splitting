from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors.torch import load_file
import torch.nn as nn
import os
import time

from config import (
    DEVICE
)

def setup_model_a(stopping_layer:int, model_path, prompt):
    start = time.time()
    model_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    config.num_hidden_layers = stopping_layer

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    model_name = os.path.basename(model_path)
    layers_dir = f"./layers/{model_name}"
    state_a = {}

    state_a.update(load_file(f"{layers_dir}/embed_tokens.safetensors", device=DEVICE))
    for i in range(stopping_layer):
        state_a.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=DEVICE))
        print(f"Loaded layer {i}")

    model.load_state_dict(
        state_a,
        strict=False,
        assign=True
    )

    model.eval()

    # Prompt setup lives on Machine A — it drives the generation loop
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print(f"Load time: {time.time() - start:.2f}s \n")
    print("Machine A ready \n")

    return model, inputs, tokenizer

def setup_model_b(stopping_layer:int, model_path):
    start = time.time()

    model_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    original_total_layers = config.num_hidden_layers 

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model_name = os.path.basename(model_path)
    layers_dir = f"./layers/{model_name}"
    state_b = {}

    for i in range(stopping_layer, original_total_layers):
        state_b.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=DEVICE))
        print(f"Loaded layer {i}")

    state_b.update(load_file(f"{layers_dir}/norm.safetensors", device=DEVICE))
    state_b.update(load_file(f"{layers_dir}/head.safetensors", device=DEVICE))

    model.load_state_dict(
        state_b,
        strict=False,
        assign=True
    )
    

    kept_layers = model.model.layers[stopping_layer:]
    model.model.layers = nn.ModuleList(kept_layers)
    
    #for i, layer in enumerate(model.model.layers):
        #print(i, layer.input_layernorm.weight.device)

    model.eval()

    print(f"Load time: {time.time() - start:.2f}s \n")
    print("Machine B ready \n")

    return model, tokenizer