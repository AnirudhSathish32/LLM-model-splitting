from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer
import torch
import time
import os
import psutil
import socket
import io

model_path = "./llama-3b"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
stopping_layer = 14
starting_layer = stopping_layer + 1
tokens_to_generate = 200
first_pass = True 

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"}
)

model.model.layers = torch.nn.ModuleList(
    model.model.layers[14:]
)

model.eval()

MACHINE_A_TAILSCALE_IP = "100.74.100.92"  
TAILSCALE_PORT = 65432

MSG_FIRST_PASS = 1
MSG_NEXT_PASS = 2
MSG_TOKEN = 3
MSG_EOS = 4

def send_token(conn, token):
    buffer = io.BytesIO()
    torch.save(token, buffer)
    payload = buffer.getvalue()
    conn.sendall(MSG_TOKEN.to_bytes(1, byteorder="big"))
    conn.sendall(len(payload).to_bytes(8, byteorder="big"))
    conn.sendall(payload)
    print(f"Token sent to Machine A")

def send_eos(conn):
    conn.sendall(MSG_EOS.to_bytes(1, byteorder="big"))
    conn.sendall((0).to_bytes(8, byteorder="big"))
    print("EOS sent to Machine A")

def receive_msg_file(conn, expected_msg_type, save_path):
    msg_type = read_TCP_data(conn, 1)[0]
    if msg_type != expected_msg_type:
        raise ValueError(
            f"Expected msg {expected_msg_type}, got {msg_type}"
        )
    receive_file(conn, save_path)


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

def setup_machine_b(retries=20, delay=3):
    print(f"Machine B connecting to {MACHINE_A_TAILSCALE_IP}:{TAILSCALE_PORT}")
    for attempt in range(1, retries + 1):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Create client socket
            # AF_INET = IPv4 addressing
            # SOCK_STREAM means TCP
            client_socket.connect((MACHINE_A_TAILSCALE_IP, TAILSCALE_PORT))
            # Attempts TCP handshake
            print(f"Connected to Machine A on attempt {attempt}")
            return client_socket
        except ConnectionRefusedError:
            print(f"Attempt {attempt}/{retries} — Machine A not ready, retrying in {delay}s...")
            client_socket.close()
            time.sleep(delay)
    raise ConnectionError("Could not connect to Machine A")

def receive_file(conn, save_path):
    
    length = int.from_bytes(read_TCP_data(conn, 8), byteorder="big")
    # read exactly the first 8 bytes which contain the file size
    # int.from_bytes = turn bytes back into numbers

    print(f"Receiving {length} bytes...")
    
    data = read_TCP_data(conn, length)
    # read the payload
    
    with open(save_path, "wb") as f:
    # open destination file in binary write mode
        f.write(data)
        # write the data
    print(f"File saved to {save_path}")

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

def load_handoff_package(save_dir="./received", first_pass=True):
    device = "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    """
    if first_pass:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=device)
        cos = torch.load(f"{save_dir}/cos.pt", map_location=device)
        sin = torch.load(f"{save_dir}/sin.pt", map_location=device)
        position_embeddings = (cos, sin)
        position_ids = torch.load(f"{save_dir}/position_ids.pt", map_location=device)
        return hidden, position_embeddings, position_ids
    else:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=device)
        return hidden

def split_2(hidden, position_embeddings, position_ids, cache_b=None):
    """
    ---- Machine B ----
    Second Split 
    """
    
    if cache_b is None:
        cache_b = DynamicCache()
        for _ in range(len(model.model.layers)):
            cache_b.layers.append(DynamicLayer())

    with torch.no_grad():
        x = hidden

        for i in range(len(model.model.layers)):
            x = model.model.layers[i](
                x,
                position_ids= position_ids,
                position_embeddings=position_embeddings,
                past_key_value=cache_b.layers[i]
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
    while True:
        
        if first_pass:
            print("Machine B first pass")
            os.makedirs("./received", exist_ok=True)
            receive_msg_file(conn, MSG_FIRST_PASS,"./received/hidden.pt")
            receive_msg_file(conn, MSG_FIRST_PASS,"./received/sin.pt")
            receive_msg_file(conn, MSG_FIRST_PASS,"./received/position_ids.pt")
            receive_msg_file(conn, MSG_FIRST_PASS,"./received/cos.pt")

            hidden, position_embeddings, position_ids = load_handoff_package(first_pass=first_pass)
            first_pass = False
            #load file into memory

        else:
            receive_msg_file(conn, MSG_NEXT_PASS,"./received/hidden.pt")
            receive_msg_file(conn, MSG_NEXT_PASS,"./received/sin.pt")
            receive_msg_file(conn, MSG_NEXT_PASS,"./received/position_ids.pt")
            receive_msg_file(conn, MSG_NEXT_PASS,"./received/cos.pt")
            hidden, position_embeddings, position_ids = load_handoff_package()

        print(f"hidden device: {hidden.device}")
        print(f"position_ids device: {position_ids.device}")
        print(f"cos device: {position_embeddings[0].device}")
        print(f"sin device: {position_embeddings[1].device}")

        print("Starting Split 2")
        next_token_id, cache_b = split_2(hidden, position_embeddings, position_ids, cache_b)
        #perform split 2 and generate the next token

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        if next_token_id.item() in eos_ids:
            # if we have detect eos/reached token count then we call machine A to start decoding the response by sending eos_detected = True
            eos_detected = True
            print("sending eos")
            send_eos(conn)
            break
        else:
            print("sending token")
            send_token(conn, next_token_id)

    get_system_stats("==================== SPLIT GEN STATS ============================")
    return


if __name__ == "__main__":
    conn = setup_machine_b()
    try:
        run_machine_b(conn)
    finally:
        conn.close()