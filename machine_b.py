from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer, AutoConfig
from accelerate import init_empty_weights
from safetensors.torch import load_file
import time
import os

from config import (
    MODEL_PATH,
    STOPPING_LAYER,
    MSG_FIRST_PASS,
    MSG_NEXT_PASS,
    TOKENS_TO_GENERATE,
    RECEIVED_DIR
)
from networking import (

    setup_machine_b_conn,
    receive_msg_file,
    receive_ttft,
    send_token,
    send_eos,
    send_layers,
    receive_layers
)

from inferencing import (
    load_handoff_package,
    split_2
)

from model_loading import (
    setup_model_b
)


def run_machine_b(tokenizer, model, stopping_layer, tokens_to_generate, conn):
    generated_token_ids = []
    cache_b = None
    position_embeddings = None
    position_ids = None
    first_pass = True
    token_count = 0 
    eos_detected = False

    layer_outputs_b = {}
    layer_times_b   = {}

    def make_validation_hook(idx):
        original_idx = idx + stopping_layer
        def hook_fn_validation(module, input, output):
            t = time.time()
            hidden = output[0].detach().clone()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            layer_outputs_b[original_idx] = hidden
            layer_times_b[original_idx]   = time.time() - t
        return hook_fn_validation

    # Register validation hooks on all layers
    validation_hooks = []
    for i in range(len(model.model.layers)):
        print(f"hook registered to layer {i + stopping_layer}")
        validation_hooks.append(
            model.model.layers[i].register_forward_hook(make_validation_hook(i))
        )
    
    while True:
        if first_pass:

            #for idx, tensor in layer_outputs_b.items():
                #print(f"Layer {idx} shape after first pass removal: {tensor.shape}")
            
            print("Machine B first pass")
            os.makedirs(RECEIVED_DIR, exist_ok=True)
            receive_msg_file(conn, MSG_FIRST_PASS, f"{RECEIVED_DIR}/hidden.pt")
            receive_msg_file(conn, MSG_FIRST_PASS, f"{RECEIVED_DIR}/sin.pt")
            receive_msg_file(conn, MSG_FIRST_PASS, f"{RECEIVED_DIR}/position_ids.pt")
            receive_msg_file(conn, MSG_FIRST_PASS, f"{RECEIVED_DIR}/cos.pt")

            hidden, position_embeddings, position_ids = load_handoff_package(first_pass=first_pass)
            first_pass = False
            #load file into memory

        else:
            receive_msg_file(conn, MSG_NEXT_PASS, f"{RECEIVED_DIR}/hidden.pt")
            receive_msg_file(conn, MSG_NEXT_PASS, f"{RECEIVED_DIR}/sin.pt")
            receive_msg_file(conn, MSG_NEXT_PASS, f"{RECEIVED_DIR}/position_ids.pt")
            receive_msg_file(conn, MSG_NEXT_PASS, f"{RECEIVED_DIR}/cos.pt")
            hidden, position_embeddings, position_ids = load_handoff_package()


        print(f"Starting Split 2: Pass #{token_count + 1}")
        next_token_id, cache_b = split_2(hidden, position_embeddings, position_ids, model, cache_b)
        #print(hidden.dtype, hidden.device)
        for h in validation_hooks:
                h.remove()
        validation_hooks = []
        #perform split 2 and generate the next token

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        if next_token_id.item() in eos_ids:
            # if we have detect eos/reached token count then we call machine A to start decoding the response by sending eos_detected = True
            eos_detected = True
            send_eos(conn)
            print("Sent EOS Token")
            break

        else:
            generated_token_ids.append(next_token_id.item())
            send_token(conn, next_token_id)
            token_count += 1
            print(f"Sent Token {token_count} \n")
            if token_count >= tokens_to_generate:
                break

    #print(f"layer_outputs_b keys before send: {sorted(layer_outputs_b.keys())}")
    #print(f"layer_outputs_b length: {len(layer_outputs_b)}")

    print("Receiving Machine A layer outputs...")
    machine_a_layer_outputs = receive_layers(conn)

    print("Sending Machine B layer outputs to Machine A...")
    send_layers(conn, layer_outputs_b)

    all_layer_outputs = {**machine_a_layer_outputs, **layer_outputs_b}
    print(len(all_layer_outputs))

    ttft = receive_ttft(conn)
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    #get_system_stats("==================== SPLIT GEN STATS ============================")
    return response, all_layer_outputs, ttft

# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    conn = setup_machine_b_conn()
    model, tokenizer = setup_model_b(STOPPING_LAYER, MODEL_PATH)
    try:
        response, all_layer_outputs, ttft = run_machine_b(tokenizer, model, STOPPING_LAYER, TOKENS_TO_GENERATE, conn)
        print("response:", response)
    finally:
        conn.close()