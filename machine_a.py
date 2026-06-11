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
    DEBUG,
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
    send_handoff,
    send_msg_file,
    send_ttft,
    send_layers,
    receive_layers,
    receive_response
)

from inferencing import (
    save_handoff_package,
    split_1
)

from hooks import (
    make_layer_hook,
    positional_hook,
    layer_hooks,
    layer_history,
    handoff_package
)

from model_loading import (
    setup_model_a
)

def run_machine_a(tokens_to_generate, stopping_layer, tokenizer, inputs, model, conn):
    generated_token_ids = []
    full_sequence_ids = inputs["input_ids"]
    cache_a = None
    position_embeddings = None
    first_pass = True
    boundary = stopping_layer - 1
    pass_counter = {"i": 0}
    token_count = 0 
    ttft = None
    ttft_start = time.perf_counter()

    for i in range(len(model.model.layers)):
        if DEBUG or i == boundary:
            pre_timer, hidden_hook = make_layer_hook(boundary, pass_counter, i)
            layer = model.model.layers[i]
            layer_hooks[i] = (
                layer.register_forward_pre_hook(pre_timer, with_kwargs=True),
                layer.register_forward_hook(hidden_hook),
            )
        
    position_hook = model.model.layers[boundary].register_forward_pre_hook(positional_hook, with_kwargs=True)
    
    while token_count < tokens_to_generate:
        
        print(f"Starting Split 1: Pass #{token_count + 1}")
        
        if first_pass:
            
            hidden, position_embeddings, position_ids, cache_a = split_1(full_sequence_ids, model, cache_a)
            layer_history[pass_counter["i"], stopping_layer - 1] = {
                "position_embeddings": (position_embeddings[0].detach().clone(), position_embeddings[1].detach().clone()),
                "position_ids": position_ids.detach().clone() 
            }

            # perform split 1

            if DEBUG:
                save_handoff_package(hidden, position_embeddings, position_ids)

            send_handoff(conn, MSG_FIRST_PASS, hidden, position_embeddings, position_ids)
            #print(hidden.dtype)
            first_pass = False

            #export captured["position_ids"], captured["position_embeddings"] and captured["hidden"]

        else:
            hidden, position_embeddings, position_ids, cache_a = split_1(full_sequence_ids[:, -1:], model, cache_a)
            layer_history[pass_counter["i"], stopping_layer - 1] = {
                "position_embeddings": (position_embeddings[0].detach().clone(), position_embeddings[1].detach().clone()),
            }
            # perform split 1
            if DEBUG:
                save_handoff_package(hidden, position_embeddings, position_ids)
            
            send_handoff(conn, MSG_NEXT_PASS, hidden, position_embeddings, position_ids=None)

        
        # call machine_b
        msg_string, token = receive_response(conn)
        
        if ttft is None:
            ttft = time.perf_counter() - ttft_start

        if msg_string == "eos":
            print("received EOS")
            break

        generated_token_ids.append(token.item())
        full_sequence_ids = torch.cat([full_sequence_ids, token.unsqueeze(0).to(full_sequence_ids.device)], dim=-1)
        token_count += 1
        pass_counter["i"] += 1
        print(f"received token {token_count} \n")
    all_layer_history = {}

    if DEBUG:
        print("Sending Machine A layer outputs to Machine B...")
        send_layers(conn, layer_history)
        print("Receiving Machine B layer outputs...")
        machine_b_layer_history = receive_layers(conn)
        print("Sending ttft to Machine B")
        

        all_layer_history = {**layer_history, **machine_b_layer_history}
        
    for handles in layer_hooks.values():
            for h in handles:
                h.remove()
    position_hook.remove()
    send_ttft(conn, ttft)
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return response, all_layer_history, ttft


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    server_socket, conn = setup_machine_a_conn()
    model, inputs, tokenizer = setup_model_a(STOPPING_LAYER, MODEL_PATH, PROMPT)
    try:
        response, all_layer_history, ttft = run_machine_a(TOKENS_TO_GENERATE, STOPPING_LAYER, tokenizer, inputs, model, conn)
        print("Response:", response)
    finally:
        conn.close()
        server_socket.close()