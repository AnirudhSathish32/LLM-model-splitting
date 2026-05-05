from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
import torch
import time
import os
import socket 
import psutil
import io 

model_path = "./llama-3b"
device = "cpu"
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
    device_map=device_map
)
model.eval()

# Prompt setup lives on Machine A — it drives the generation loop
messages = [{"role": "user", "content": "hello world"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")
captured = {}
TAILSCALE_PORT = 65432

MSG_FIRST_PASS = 1
MSG_NEXT_PASS = 2
MSG_TOKEN = 3
MSG_EOS = 4

#def handle_message(msg_type, payload):
    #"""
    #    message types {1:INIT, 2:STEP, 3:XXXX}
    #"""
    #message_types = {1:"FIRST_PASS", 2:"NEXT_PASS", 3:"TOKEN", 4:"EOS"}
    #msg_name = message_types.get(msg_type)




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

def run_machine_a(tokens_to_generate, conn):
    generated_token_ids = []
    
    current_input_ids = inputs["input_ids"]
    
    cache_a = None
    position_embeddings = None
    position_ids = None
    first_pass = True
    token_count = 0 

    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)

    while token_count < tokens_to_generate:
        
        hidden, position_embeddings, position_ids, cache_a = split_1(current_input_ids, cache_a)
        # perform split 1
        
        if first_pass:
            save_handoff_package(hidden, position_embeddings, position_ids)

            conn.sendall(MSG_FIRST_PASS.to_bytes(1, byteorder="big"))

            send_to_machine_b(conn, "./handoff/hidden.pt")
            send_to_machine_b(conn, "./handoff/sin.pt")
            send_to_machine_b(conn, "./handoff/position_ids.pt")
            send_to_machine_b(conn, "./handoff/cos.pt")
            first_pass = False

            #export captured["position_ids"], captured["position_embeddings"] and captured["hidden"]

        else:
            save_handoff_package(hidden)

            conn.sendall(MSG_NEXT_PASS.to_bytes(1, byteorder="big"))

            send_to_machine_b(conn, "./handoff/hidden.pt")


        # call machine_b
        msg_type, payload = read_message(conn)

        if msg_type == MSG_EOS:
            break

        if msg_type == MSG_TOKEN:
            print("receiving token")
            next_token_id = torch.load(io.BytesIO(payload))
            generated_token_ids.append(next_token_id.item())
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)
            token += 1

    h1.remove()
    h2.remove()
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    return response

if __name__ == "__main__":
    server_socket, conn = setup_machine_a_conn()
    try:
        result = run_machine_a(tokens_to_generate, conn)
        print("Response:", result)
    finally:
        conn.close()
        server_socket.close()