# ============================================================
# IMPORTS / GLOBAL CONFIG
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer, AutoConfig
from accelerate import init_empty_weights
from safetensors.torch import load_file
import torch
import time
import os
import socket 
import psutil
import io 
import struct
from config import (
    MODEL_PATH,
    STOPPING_LAYER,
    PROMPT,
    TOKENS_TO_GENERATE,
    DEVICE,
    TAILSCALE_PORT,
    MSG_FIRST_PASS,
    MSG_NEXT_PASS,
    MSG_TOKEN,
    MSG_EOS,
    MSG_LAYER,
    MSG_TTFT,
    HANDOFF_DIR
)

# ============================================================
# MODEL LOADING / INITIALIZATION
# ============================================================

captured = {}

def setup_model_a(stopping_layer:int, model_path, prompt):
    start = time.time()
    model_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    config.num_hidden_layers = stopping_layer

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    model_name = os.path.basename(model_path)
    layers_dir = f"./layers/{model_name}"
    state_a = {}

    state_a.update(load_file(f"{layers_dir}/embed_tokens.safetensors", device=DEVICE))
    for i in range(stopping_layer):
        state_a.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=DEVICE))
        print(f"Loaded layer {i}")

    model.load_state_dict(
        state_a,
        strict=False,
        assign=True
    )

    model.eval()

    # Prompt setup lives on Machine A — it drives the generation loop
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print(f"Load time: {time.time() - start:.2f}s")
    print("Machine A ready")

    return model, inputs, tokenizer

# ============================================================
# MESSAGE PROTOCOL / SOCKET COMMUNICATION
# ============================================================

def send_ttft(conn, ttft):
    """
    Send Time-To-First-Token as an 8-byte float
    """

    payload = struct.pack(">d", ttft)
    # >d:
    # > = big endian
    # d = double precision float (8 bytes)

    conn.sendall(bytes([MSG_TTFT]))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)

    print(f"Sent TTFT: {ttft:.4f}s")

def send_layers(conn, layers):
    buffer = io.BytesIO()
    torch.save(layers, buffer)
    payload = buffer.getvalue()
    conn.sendall(MSG_LAYER.to_bytes(1, byteorder="big"))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)
    print(f"Layers sent to Machine B")

def receive_layers(conn):
    msg_type, payload = read_message(conn)
    if msg_type != MSG_LAYER:
        raise ValueError(
            f"Expected MSG_LAYER, got {msg_type}"
        )
    return torch.load(io.BytesIO(payload), map_location=DEVICE)

#def handle_message(msg_type, payload):
    #"""
    #    message types {1:INIT, 2:STEP, 3:XXXX}
    #"""
    #message_types = {1:"FIRST_PASS", 2:"NEXT_PASS", 3:"TOKEN", 4:"EOS"}
    #msg_name = message_types.get(msg_type)

def send_msg_file(conn, msg_type, filepath):
    conn.sendall(msg_type.to_bytes(1, "big"))
    send_to_machine_b(conn, filepath)


def read_message(conn):
    msg_type = read_TCP_data(conn, 1)[0] 
    length = int.from_bytes(read_TCP_data(conn, 8), "big") 
    payload = read_TCP_data(conn, length) 
    return msg_type, payload

def read_TCP_data(conn, length):
    """
        helper function

        conn = TCP socket connection between Machine A and B brokered by Tailscale
        length = exact number of bytes expected in the incoming data

        returns data in binary format
    
    """
    data = b""
    # empty bytes buffer, this is raw binary data

    while len(data) < length:
        # we loop until we have enough bytes collected
        packet = conn.recv(length - len(data))
        # the packet = length needed - length of data currently being processed 
        if not packet:
            raise ConnectionError("Connection dropped")
        data += packet
        # add packet binaries to data
    return data

def send_to_machine_b(conn, filepath):
    with open(filepath, "rb") as f:
        # Open file in binary read mode
        # tensor files contain raw serialized bytes so text would corrupt the data
        data = f.read()
        # load file into memory
    conn.sendall(len(data).to_bytes(8, byteorder="big"))
    # len(data).tobytes(8) = let the first 8 bytes = the file length
    # byteorder = big = send the most siginificant byte first
    # we are telling the receiver how much data is coming 
    conn.sendall(data)
    # sending the actual data
    print(f"Sent {filepath} ({len(data)} bytes)")

def setup_machine_a_conn():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Create server socket
    # AF_INET = IPv4 addressing
    # SOCK_STREAM means TCP
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Allows for the reuse of the port immediately after the program exits
    server_socket.bind(("0.0.0.0", TAILSCALE_PORT))
    # Listen across all network interfaces on the TAILSCALE port
    server_socket.listen(1)
    # backlog size = 1, waiting for incoming connections
    print(f"Machine A listening on port {TAILSCALE_PORT}...")
    conn, addr = server_socket.accept()
    # when Machine B connects we return conn and addr
    print(f"Machine B connected from {addr}")
    return server_socket, conn

# ============================================================
# VALIDATION / BENCHMARKING
# ============================================================

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

# ============================================================
# FORWARD HOOK CAPTURE FUNCTIONS
# ============================================================

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

def save_handoff_package(hidden, position_embeddings, position_ids, save_dir=HANDOFF_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden, f"{save_dir}/hidden.pt")
    torch.save(position_embeddings[0], f"{save_dir}/cos.pt")
    torch.save(position_embeddings[1], f"{save_dir}/sin.pt")
    torch.save(position_ids, f"{save_dir}/position_ids.pt")

def save_hidden_only(hidden, save_dir=HANDOFF_DIR):
    torch.save(hidden, f"{save_dir}/hidden.pt")

# ============================================================
# SPLIT EXECUTION (MACHINE A)
# ============================================================

def split_1(current_input_ids, model, cache_a=None):
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

def run_machine_a(tokens_to_generate, stopping_layer, tokenizer, inputs, model, conn):
    generated_token_ids = []
    current_input_ids = inputs["input_ids"]
    cache_a = None
    position_embeddings = None
    position_ids = None
    first_pass = True
    token_count = 0 

    ttft_start  = time.time()
    ttft        = None
    layer_outputs = {}
    layer_times   = {}
    ttft_result   = {"ttft": None, "start": time.time(), "fired": False}

    def make_validation_hook(idx):
        def hook_fn_validation(module, input, output):
            t = time.time()

            if idx == 0 and not ttft_result["fired"]:
                ttft_result["ttft"]  = t - ttft_result["start"]
                ttft_result["fired"] = True

            hidden = output[0].detach().clone()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            layer_outputs[idx] = hidden
            layer_times[idx]   = time.time() - t
        return hook_fn_validation

    # Register validation hooks on all layers
    validation_hooks = []
    for i in range(len(model.model.layers)):
        validation_hooks.append(
            model.model.layers[i].register_forward_hook(make_validation_hook(i))
        )



    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)

    while token_count < tokens_to_generate:
        
        print("Starting Split 1")
        hidden, position_embeddings, position_ids, cache_a = split_1(current_input_ids, model, cache_a)
        # perform split 1
        
        if first_pass:

            for h in validation_hooks:
                h.remove()
            validation_hooks = []

            #for idx, tensor in layer_outputs.items():
                #print(f"Layer {idx} shape after first pass removal: {tensor.shape}")

            save_handoff_package(hidden, position_embeddings, position_ids)

            send_msg_file(conn, MSG_FIRST_PASS, f"{HANDOFF_DIR}/hidden.pt")
            send_msg_file(conn, MSG_FIRST_PASS, f"{HANDOFF_DIR}/sin.pt")
            send_msg_file(conn, MSG_FIRST_PASS, f"{HANDOFF_DIR}/position_ids.pt")
            send_msg_file(conn, MSG_FIRST_PASS, f"{HANDOFF_DIR}/cos.pt")
            print(hidden.dtype)
            first_pass = False

            #export captured["position_ids"], captured["position_embeddings"] and captured["hidden"]

        else:
            save_handoff_package(hidden, position_embeddings, position_ids)
            send_msg_file(conn, MSG_NEXT_PASS, f"{HANDOFF_DIR}/hidden.pt")
            send_msg_file(conn, MSG_NEXT_PASS, f"{HANDOFF_DIR}/sin.pt")
            send_msg_file(conn, MSG_NEXT_PASS, f"{HANDOFF_DIR}/position_ids.pt")
            send_msg_file(conn, MSG_NEXT_PASS, f"{HANDOFF_DIR}/cos.pt")


        # call machine_b
        msg_type, payload = read_message(conn)

        if ttft is None:
            ttft = time.time() - ttft_start
            print(f"\n--- First Pass Validation Capture ---")
            print(f"Layers captured:     {len(layer_outputs)}")
            print(f"Time to first token: {ttft:.3f}s")
            print(f"{'Layer':<8} {'Shape':<25} {'Time (ms)':<12}")
            print(f"{'-'*45}")
            for idx in sorted(layer_outputs.keys()):
                shape   = str(tuple(layer_outputs[idx].shape))
                elapsed = layer_times.get(idx, 0) * 1000
                print(f"{idx:<8} {shape:<25} {elapsed:<12.3f}")
            print(f"{'-'*45}\n")

        if msg_type == MSG_EOS:
            print("received EOS")
            break

        if msg_type == MSG_TOKEN:
            print("receiving token")
            next_token_id = torch.load(io.BytesIO(payload))
            generated_token_ids.append(next_token_id.item())
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).to(current_input_ids.device)], dim=-1)
            token_count += 1
            print(token_count)

    print("Sending Machine A layer outputs to Machine B...")
    send_layers(conn, layer_outputs)
    print("Receiving Machine B layer outputs...")
    machine_b_layer_outputs = receive_layers(conn)
    print("Sending ttft to Machine B")
    ttft = ttft_result["ttft"]
    send_ttft(ttft)

    h1.remove()
    h2.remove()
    all_layer_outputs = {**layer_outputs, **machine_b_layer_outputs}
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return response, all_layer_outputs, ttft


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    server_socket, conn = setup_machine_a_conn()
    model, inputs, tokenizer = setup_model_a(STOPPING_LAYER, MODEL_PATH, PROMPT)
    try:
        response, all_layer_outputs, ttft = run_machine_a(TOKENS_TO_GENERATE, STOPPING_LAYER, tokenizer, inputs, model, conn)
        print("Response:", response)
    finally:
        conn.close()
        server_socket.close()