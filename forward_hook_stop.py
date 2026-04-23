from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

inputs = tokenizer("Hello world", return_tensors="pt")

layer14_output = {}

def hook_fn(module, input, output):
    layer14_output["value"] = output[0].detach().clone()
    raise StopIteration  

handle = model.model.layers[13].register_forward_hook(hook_fn)

try:
    with torch.no_grad():
        model(**inputs)
except StopIteration:
    pass

handle.remove()

print(layer14_output["value"])


