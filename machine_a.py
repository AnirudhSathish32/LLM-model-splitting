from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
import torch
import time
import os
import psutil

model_path = "./llama-3b"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
stopping_layer = 14
starting_layer = stopping_layer + 1
tokens_to_generate = 200

# Machine A device map — only loads layers 0-13
device_map = {"model.embed_tokens": device}
for i in range(28):
    device_map[f"model.layers.{i}"] = device if i < stopping_layer else "meta"
device_map["model.norm"] = "meta"
device_map["lm_head"] = "meta"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=device_map
)
model.eval()

# Prompt setup lives on Machine A — it drives the generation loop
messages = [{"role": "user", "content": "Best Alfredo pasta recipe"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

captured = {}

def get_system_stats(label):
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # RAM usage
    ram = psutil.Process(os.getpid()).memory_info().rss / 1e9
    
    print(f"\n--- {label} ---")
    print(f"CPU usage:    {cpu_percent:.1f}%")
    print(f"RAM usage:    {ram:.2f} GB")
    
    # GPU stats if available
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved  = torch.cuda.memory_reserved() / 1e9
        gpu_total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU allocated: {gpu_allocated:.2f} GB")
        print(f"GPU reserved:  {gpu_reserved:.2f} GB")
        print(f"GPU total:     {gpu_total:.2f} GB")
    else:
        print("GPU: not available")

def hook_fn(module, input, output):
        hidden = output[0].detach()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured["hidden"] = hidden
        raise StopIteration
    
def hook_pos(module, args, kwargs):
    cos, sin = kwargs.get("position_embeddings")
    captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
    captured["position_ids"] = kwargs.get("position_ids")
    captured["cache_a"] = kwargs.get("past_key_value")

def save_handoff_package(hidden, position_embeddings, position_ids, save_dir="./handoff"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden, f"{save_dir}/hidden.pt")
    torch.save(position_embeddings[0], f"{save_dir}/cos.pt")
    torch.save(position_embeddings[1], f"{save_dir}/sin.pt")
    torch.save(position_ids, f"{save_dir}/position_ids.pt")

def split_1(current_input_ids, cache_a=None):
    """
    ---- Machine A ----
    First Split

    """
    try:
        with torch.no_grad():
            model(input_ids=current_input_ids,
                past_key_values=cache_a,
                use_cache=True,
                return_dict=True)
    except StopIteration:
        pass
    hidden = captured["hidden"]
    position_embeddings = captured["position_embeddings"]
    position_ids = captured["position_ids"]
    cache_a = captured["cache_a"]

    return hidden, position_embeddings, position_ids, cache_a

def perform_split_generation(tokens_to_generate):
    generated_token_ids = []
    
    # Start with the original input ids
    current_input_ids = inputs["input_ids"]
    
    cache_a = None
    position_embeddings = None
    position_ids = None
    first_pass = True
    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)
    for _ in range(tokens_to_generate):
        
        hidden, position_embeddings, position_ids, cache_a = split_1(current_input_ids, cache_a)
        # perform split 1
        
        if first_pass:
            save_handoff_package(hidden, position_embeddings, position_ids)
            #export captured["position_ids"], captured["position_embeddings"] and captured["hidden"]

        else:
            save_handoff_package(hidden)


        # call machine_b
        # next_token_id = split_2(hidden, position_embeddings, position_ids, cache_b)
        #perform split 2 and generate the next token