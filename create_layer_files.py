from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import torch
import os

model_path = "./llama-3b"
output_dir = "./layers"

model_name = os.path.basename(model_path)
layers_dir = f"{output_dir}/{model_name}"
os.makedirs(layers_dir, exist_ok=True)

print("Loading full model...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
full_state = model.state_dict()

# Save each layer separately
for layer_idx in range(28):
    layer_weights = {}
    prefix = f"model.layers.{layer_idx}."
    for name, tensor in full_state.items():
        if name.startswith(prefix):
            layer_weights[name] = tensor.contiguous()
    save_file(layer_weights, f"{layers_dir}/layer_{layer_idx}.safetensors")
    print(f"Saved layer {layer_idx}")

# Save embeddings separately
embed_weights = {k: v.contiguous() for k, v in full_state.items() 
                 if k.startswith("model.embed_tokens")}
save_file(embed_weights, f"{layers_dir}/embed_tokens.safetensors")
print("Saved embed_tokens")

# Save norm and lm_head separately
head_weights = {k: v.contiguous() for k, v in full_state.items() 
                if k.startswith("lm_head")}
save_file(head_weights, f"{layers_dir}/head.safetensors")
print("Saved head")

norm_weights = {k: v.contiguous() for k, v in full_state.items() 
                if k.startswith("model.norm")}
save_file(norm_weights, f"{layers_dir}/norm.safetensors")
print("Saved norm")

print("Done — per layer files created")