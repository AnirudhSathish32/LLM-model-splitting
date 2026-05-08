from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import torch
import os

model_path = "./llama-3b"
output_dir = "./layers"
os.makedirs(output_dir, exist_ok=True)

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
    save_file(layer_weights, f"{output_dir}/layer_{layer_idx}.safetensors")
    print(f"Saved layer {layer_idx}")

# Save embeddings separately
embed_weights = {k: v.contiguous() for k, v in full_state.items() 
                 if k.startswith("model.embed_tokens")}
save_file(embed_weights, f"{output_dir}/embed_tokens.safetensors")
print("Saved embed_tokens")

# Save norm and lm_head separately
head_weights = {k: v.contiguous() for k, v in full_state.items() 
                if k.startswith("model.norm") or k.startswith("lm_head")}
save_file(head_weights, f"{output_dir}/head.safetensors")
print("Saved head")

print("Done — per layer files created")