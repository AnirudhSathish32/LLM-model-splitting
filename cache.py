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

inputs = tokenizer("Hello world", return_tensors="pt")

# Run one forward pass with cache enabled
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        use_cache=True,
        return_dict=True
    )

cache = output.past_key_values

from transformers import DynamicCache

def split_cache(cache, split_layer=14):
    # Machine A cache — layers 0 to 13
    cache_a = DynamicCache()
    for layer in cache.layers[:split_layer]:
        cache_a.layers.append(layer)

    # Machine B cache — layers 14 to 27
    cache_b = DynamicCache()
    for layer in cache.layers[split_layer:]:
        cache_b.layers.append(layer)

    print(f"Cache A layers: {len(cache_a.layers)}")
    print(f"Cache B layers: {len(cache_b.layers)}")
    print(f"Cache A layer 0 keys: {cache_a.layers[0].keys.shape}")
    print(f"Cache B layer 0 keys: {cache_b.layers[0].keys.shape}")

    return cache_a, cache_b

cache_a, cache_b = split_cache(cache)



print(f"Number of layers: {len(cache.layers)}")
print(f"Layer 0 type: {type(cache.layers[0])}")
print(f"Layer 0 keys shape:   {cache.layers[0].keys.shape}")
print(f"Layer 0 values shape: {cache.layers[0].values.shape}")

# Print all layers
for i, layer in enumerate(cache.layers):
    print(f"Layer {i}: keys={layer.keys.shape} values={layer.values.shape}")

