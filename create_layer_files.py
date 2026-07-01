"""
create_layer_files.py

One-time utility: split a HuggingFace model into per-layer safetensors.

Usage:
    python create_layer_files.py <model_name>
    
    Reads model from <model_path>/<model_name>/ (per LocalConfig)
    Writes layers to <layers_path>/<model_name>/
"""

import sys, os, torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from config import LocalConfig

if len(sys.argv) < 2:
    print("Usage: python create_layer_files.py <model_name>")
    sys.exit(1)

model_name = sys.argv[1]
local = LocalConfig.load()

model_path = os.path.join(local.model_path, model_name)
layers_dir = os.path.join(local.layers_path, model_name)
os.makedirs(layers_dir, exist_ok=True)

print(f"Loading full model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
full_state = model.state_dict()

# Save each layer separately
for layer_idx in range(len(model.model.layers)):
    layer_weights = {}
    prefix = f"model.layers.{layer_idx}."
    for name, tensor in full_state.items():
        if name.startswith(prefix):
            layer_weights[name] = tensor.contiguous()
    save_file(layer_weights, f"{layers_dir}/layer_{layer_idx}.safetensors")
    print(f"  Saved layer {layer_idx}")

# Save embeddings
embed_weights = {k: v.contiguous() for k, v in full_state.items()
                 if k.startswith("model.embed_tokens")}
save_file(embed_weights, f"{layers_dir}/embed_tokens.safetensors")
print("  Saved embed_tokens")

# Save lm_head
head_weights = {k: v.contiguous() for k, v in full_state.items()
                if k.startswith("lm_head")}
save_file(head_weights, f"{layers_dir}/head.safetensors")
print("  Saved head")

# Save norm
norm_weights = {k: v.contiguous() for k, v in full_state.items()
                if k.startswith("model.norm")}
save_file(norm_weights, f"{layers_dir}/norm.safetensors")
print("  Saved norm")

print(f"\nDone — {len(model.model.layers)} layer files created in {layers_dir}")
