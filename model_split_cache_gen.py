from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
import torch
import time
import os
import psutil


model_path = "./llama-8b"

device = "cpu"
"""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
"""
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device)
#Loading model 
tokenizer = AutoTokenizer.from_pretrained(model_path)
#Loading Tokenizer
messages = [
    {"role": "user", "content": "Best Alfredo pasta recipe"}
]
# Loading context
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

#Loading prompt
inputs = tokenizer(prompt, return_tensors="pt")
#Tokenizing prompt into input tensors
model.eval()
#neural network enters evaluation mode so it behaves predictably

stopping_layer = 14
starting_layer = stopping_layer + 1
tokens_to_generate = 200
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

def load_handoff_package(save_dir="./handoff", first_pass=True):
    device = "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    """
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


def perform_full_generation():
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    get_system_stats("================= FULL GEN STATS ======================")
    return print(response)

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

def perform_split_generation(tokens_to_generate):
    generated_token_ids = []
    
    # Start with the original input ids
    current_input_ids = inputs["input_ids"]
    
    cache_a = None
    cache_b = None
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

            hidden, position_embeddings, position_ids = load_handoff_package(first_pass=first_pass)
            #load file into memory

        else:
            save_handoff_package(hidden)
            hidden = load_handoff_package(first_pass)

        next_token_id, cache_b = split_2(hidden, position_embeddings, position_ids, cache_b)
        #perform split 2 and generate the next token

        generated_token_ids.append(next_token_id.item())
        #Add this to the generated tokens list

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if next_token_id.item() in eos_ids:
            break
        
        current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)
        #Append new token to input for next pass

        #next step when len(current_input_ids) increases we call machine A to run its half  
    h1.remove()
    h2.remove()
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    get_system_stats("==================== SPLIT GEN STATS ============================")
    return print(response)

if __name__ == "__main__":
    start = time.time()
    print("=====================FULL GENERATION============================")
    perform_full_generation()
    print("=====================SPLIT GENERATION============================")
    perform_split_generation(tokens_to_generate)
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")