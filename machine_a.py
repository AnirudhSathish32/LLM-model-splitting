# ============================================================
# IMPORTS / GLOBAL CONFIG
# ============================================================

import torch
import time
import io 
from config import (
    MODEL_PATH,
    STOPPING_LAYER,
    PROMPT,
    TOKENS_TO_GENERATE,
    MSG_FIRST_PASS,
    MSG_NEXT_PASS,
    MSG_TOKEN,
    MSG_EOS,
    HANDOFF_DIR
)

from networking import (
    setup_machine_a_conn,
    read_message,
    send_msg_file,
    send_ttft,
    send_layers,
    receive_layers,

)

from inferencing import (
    save_handoff_package,
    split_1
)

from hooks import (
    hook_fn,
    hook_pos
)

from model_loading import (
    setup_model_a
)

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
        print(f"hook registered to layer {i}")
        validation_hooks.append(
            model.model.layers[i].register_forward_hook(make_validation_hook(i))
        )



    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)

    while token_count < tokens_to_generate:
        
        print(f"Starting Split 1: Pass #{token_count + 1}")
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
            #print(hidden.dtype)
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
            next_token_id = torch.load(io.BytesIO(payload))
            generated_token_ids.append(next_token_id.item())
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).to(current_input_ids.device)], dim=-1)
            token_count += 1
            print(f"received token {token_count} \n")

    print("Sending Machine A layer outputs to Machine B...")
    send_layers(conn, layer_outputs)
    print("Receiving Machine B layer outputs...")
    machine_b_layer_outputs = receive_layers(conn)
    print("Sending ttft to Machine B")
    send_ttft(conn, ttft)

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