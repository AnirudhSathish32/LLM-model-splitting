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
first_pass = True 

# Machine B device map — only loads layers 14-27 plus norm and lm_head
device_map = {"model.embed_tokens": "meta"}
for i in range(28):
    device_map[f"model.layers.{i}"] = device if i >= stopping_layer else "meta"
device_map["model.norm"] = device
device_map["lm_head"] = device

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=device_map
)
model.eval()

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

def load_handoff_package(save_dir="./handoff", first_pass=True):
    device = ""

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if first_pass:
        hidden = torch.load(f"{save_dir}/hidden.pt").to(device)
        cos = torch.load(f"{save_dir}/cos.pt").to(device)
        sin = torch.load(f"{save_dir}/sin.pt").to(device)
        position_embeddings = (cos, sin)
        position_ids = torch.load(f"{save_dir}/position_ids.pt").to(device)
        return hidden, position_embeddings, position_ids
    else:
        hidden = torch.load(f"{save_dir}/hidden.pt").to(device)
        return hidden

def split_2(hidden, position_embeddings, position_ids, cache_b=None):
    """
    ---- Machine B ----
    Second Split 
    """
    if cache_b is None:
        cache_b = DynamicCache()
        for _ in range(len(model.model.layers) - (starting_layer - 1)):
            cache_b.layers.append(DynamicLayer())

    with torch.no_grad():
        x = hidden

        for i in range(starting_layer - 1, len(model.model.layers)):
            cache_index = i - (starting_layer - 1)

            x = model.model.layers[i](
                x,
                position_ids= position_ids,
                position_embeddings=position_embeddings,
                past_key_value=cache_b.layers[cache_index]
            )[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)

        x = model.model.norm(x)
        logits = model.lm_head(x)

        # ---- Pick next token ----
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

    return  next_token_id, cache_b

def run_machine_b(tokens_to_generate):
    
    cache_b = None
    position_embeddings = None
    position_ids = None
    first_pass = True
    token_count = 0 
    while token_count <= tokens_to_generate:
        
        if first_pass:
            hidden, position_embeddings, position_ids = load_handoff_package(first_pass=first_pass)
            #load file into memory

        else:
            hidden = load_handoff_package(first_pass)

        next_token_id, cache_b = split_2(hidden, position_embeddings, position_ids, cache_b)
        #perform split 2 and generate the next token

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if next_token_id.item() in eos_ids:
            # if we have detect eos/reached token count then we call machine A to start decoding the response by sending eos_detected = True
            break
 
        # call machine A then return next_token_id from split 2 function

    get_system_stats("==================== SPLIT GEN STATS ============================")
    return