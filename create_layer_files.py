from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import torch
import os
from config import (
    MODEL_PATH,
    DTYPE,
    LAYERS_DIR
)

os.makedirs(LAYERS_DIR, exist_ok=True)

print("Loading full model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
full_state = model.state_dict()

# Save each layer separately
for layer_idx in range(len(model.model.layers)):
    layer_weights = {}
    prefix = f"model.layers.{layer_idx}."
    for name, tensor in full_state.items():
        if name.startswith(prefix):
            layer_weights[name] = tensor.contiguous()
    save_file(layer_weights, f"{LAYERS_DIR}/layer_{layer_idx}.safetensors")
    print(f"Saved layer {layer_idx}")

# Save embeddings separately
embed_weights = {k: v.contiguous() for k, v in full_state.items() 
                 if k.startswith("model.embed_tokens")}
save_file(embed_weights, f"{LAYERS_DIR}/embed_tokens.safetensors")
print("Saved embed_tokens")

# Save norm and lm_head separately
head_weights = {k: v.contiguous() for k, v in full_state.items() 
                if k.startswith("lm_head")}
save_file(head_weights, f"{LAYERS_DIR}/head.safetensors")
print("Saved head")

norm_weights = {k: v.contiguous() for k, v in full_state.items() 
                if k.startswith("model.norm")}
save_file(norm_weights, f"{LAYERS_DIR}/norm.safetensors")
print("Saved norm")

print("Done — per layer files created")