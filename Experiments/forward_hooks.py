from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

inputs = tokenizer("Hello world", return_tensors="pt")

# storage for captured output
layer14_output = {}

# forward hook function
def hook_fn(module, input, output):
    # output[0] = hidden states (for LLaMA layers)
    layer14_output["value"] = output[0].detach()

# attach hook to layer 14 (index 13)
handle = model.model.layers[13].register_forward_hook(hook_fn)

# run full forward pass (or generate)
with torch.no_grad():
    _ = model(**inputs)

# remove hook (important)
handle.remove()

print("Layer 14 shape:", layer14_output["value"].shape)
print(layer14_output["value"])