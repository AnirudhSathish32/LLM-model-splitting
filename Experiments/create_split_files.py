import torch
from transformers import AutoModelForCausalLM
import os

model_path = "./llama-3b"
stopping_layer = 14

# Load full model ONCE
model = AutoModelForCausalLM.from_pretrained(model_path)

full_state = model.state_dict()

machine_a = {}
machine_b = {}

for name, tensor in full_state.items():

    if name.startswith("model.embed_tokens"):
        machine_a[name] = tensor

    elif name.startswith("model.layers."):

        layer_num = int(name.split(".")[2])

        if layer_num < stopping_layer:
            machine_a[name] = tensor
        else:
            machine_b[name] = tensor

    elif name.startswith("model.norm") or name.startswith("lm_head"):
        machine_b[name] = tensor

# Save split checkpoints
os.makedirs("./split-files", exist_ok=True)
torch.save(machine_a, "./split-files/machine_a.pt")
torch.save(machine_b, "./split-files/machine_b.pt")

print("Split complete")