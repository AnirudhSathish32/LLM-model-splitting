from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"
stopping_layer = 14
starting_layer = stopping_layer + 1

tokenizer = AutoTokenizer.from_pretrained(model_path)

# ---- Load Machine A layers (0-13) ----
device_map_a = {"model.embed_tokens": "cpu"}
for i in range(28):
    device_map_a[f"model.layers.{i}"] = "cpu" if i < stopping_layer else "meta"
device_map_a["model.norm"] = "cpu"
device_map_a["lm_head"] = "cpu"

model_a = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=device_map_a
)
model_a.eval()
print("Model A loaded")

# ---- Load Machine B layers (14-27) ----
device_map_b = {"model.embed_tokens": "cpu"}
for i in range(28):
    device_map_b[f"model.layers.{i}"] = "cpu" if i >= stopping_layer else "meta"
device_map_b["model.norm"] = "cpu"
device_map_b["lm_head"] = "cpu"

model_b = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=device_map_b
)
model_b.eval()
print("Model B loaded")

# ---- Tokenize input ----
inputs = tokenizer("Hello world", return_tensors="pt")
input_ids = inputs["input_ids"]

# ---- Machine A: run layers 0-13 ----
captured = {}

def hook_fn(module, input, output):
    hidden = output[0].detach().clone()
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    captured["hidden"] = hidden
    raise StopIteration

def hook_pos(module, args, kwargs):
    cos, sin = kwargs.get("position_embeddings")
    captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
    captured["position_ids"] = kwargs.get("position_ids")

h1 = model_a.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
h2 = model_a.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)

try:
    with torch.no_grad():
        model_a(input_ids=input_ids)
except StopIteration:
    pass

h1.remove()
h2.remove()

print(f"Machine A done — hidden state shape: {captured['hidden'].shape}")

# ---- Machine B: run layers 14-27 ----
with torch.no_grad():
    x = captured["hidden"]
    pos_ids = captured["position_ids"]
    pos_emb = captured["position_embeddings"]

    for i in range(starting_layer - 1, 28):
        cache_index = i - (starting_layer - 1)
        x = model_b.model.layers[i](
            x,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
        )[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
print("machine B done")
print(x)