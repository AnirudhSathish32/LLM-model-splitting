from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

model_path = "./llama-3b"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompt = "Hello world"
inputs = tokenizer(prompt, return_tensors="pt")

# ---- Run full forward pass and get cache ----
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        use_cache=True,
        return_dict=True
    )

cache = output.past_key_values
print(f"Full cache layers: {len(cache.layers)}")

# ---- Split cache ----
def split_cache(cache, split_layer=14):
    cache_a = DynamicCache()
    for layer in cache.layers[:split_layer]:
        cache_a.layers.append(layer)

    cache_b = DynamicCache()
    for layer in cache.layers[split_layer:]:
        cache_b.layers.append(layer)

    return cache_a, cache_b

cache_a, cache_b = split_cache(cache)

print(f"Cache A layers: {len(cache_a.layers)}")
print(f"Cache B layers: {len(cache_b.layers)}")
print(f"Cache A layer 0 keys shape: {cache_a.layers[0].keys.shape}")
print(f"Cache B layer 0 keys shape: {cache_b.layers[0].keys.shape}")

# ---- Test: pass cache_b into Machine B second half ----
# First capture the hidden state at layer 14
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

h1 = model.model.layers[13].register_forward_hook(hook_fn)
h2 = model.model.layers[13].register_forward_pre_hook(hook_pos, with_kwargs=True)

try:
    with torch.no_grad():
        model(input_ids=inputs["input_ids"])
except StopIteration:
    pass

h1.remove()
h2.remove()

# ---- Run Machine B with cache_b ----
with torch.no_grad():
    x = captured["hidden"]

    for i in range(14, len(model.model.layers)):
        # Offset layer index for cache_b since it starts at 0 not 14
        cache_layer_idx = i - 14
        x = model.model.layers[i](
            x,
            position_ids=captured["position_ids"],
            position_embeddings=captured["position_embeddings"],
            past_key_value=cache_b.layers[cache_layer_idx],
        )[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)

    x = model.model.norm(x)
    logits = model.lm_head(x)

next_token = torch.argmax(logits[:, -1, :], dim=-1)
print(f"Next token: {tokenizer.decode(next_token)}")