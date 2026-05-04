from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
import torch
import time
import os
import psutil
import socket
import io

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

MACHINE_A_TAILSCALE_IP = "100.74.100.92"  # replace with Machine A's Tailscale IP
TAILSCALE_PORT = 65432

def recv_all(conn, length):
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
    return data

def setup_machine_b():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Machine B connecting to Machine A at {MACHINE_A_TAILSCALE_IP}:{TAILSCALE_PORT}...")
    client_socket.connect((MACHINE_A_TAILSCALE_IP, TAILSCALE_PORT))
    print(f"Connected to Machine A")
    return client_socket

def wait_for_machine_a(conn):
    length = int.from_bytes(recv_all(conn, 8), byteorder="big")
    data = recv_all(conn, length)
    hidden = torch.load(io.BytesIO(data))
    return hidden

def send_to_machine_a(conn, next_token_id, eos_detected):
    result = {"token": next_token_id, "eos": eos_detected}
    buffer = io.BytesIO()
    torch.save(result, buffer)
    data = buffer.getvalue()
    conn.sendall(len(data).to_bytes(8, byteorder="big"))
    conn.sendall(data)


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
    eos_detected = False
    while token_count <= tokens_to_generate:
        
        if first_pass:
            
            hidden, position_embeddings, position_ids = wait_for_machine_a(conn)

            #hidden, position_embeddings, position_ids = load_handoff_package(first_pass=first_pass)
            #load file into memory

        else:
            
            hidden = wait_for_machine_a(conn)

            #hidden = load_handoff_package(first_pass)

        next_token_id, cache_b = split_2(hidden, position_embeddings, position_ids, cache_b)
        #perform split 2 and generate the next token

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if next_token_id.item() in eos_ids:
            # if we have detect eos/reached token count then we call machine A to start decoding the response by sending eos_detected = True
            eos_detected = True

 
            # call machine A then return next_token_id from split 2 function
            send_to_machine_a(conn, next_token_id, eos_detected)
        if eos_detected:
            break

    get_system_stats("==================== SPLIT GEN STATS ============================")
    return


if __name__ == "__main__":
    conn = setup_machine_b()
    try:
        run_machine_b(conn)
    finally:
        conn.close()