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
    RECEIVED_DIR,
    DEBUG
)
from networking import (

    setup_machine_b_conn,
    receive_msg_file,
    receive_ttft,
    send_token,
    send_eos,
    send_layers,
    receive_layers,
    receive_handoff
)

from inferencing import (
    load_handoff_package,
    split_2
)

from model_loading import (
    setup_model_b,
)

from hooks import (
    make_layer_hook,
    layer_history,
    layer_hooks,
    handoff_package
)


def run_machine_b(tokenizer, model, stopping_layer, tokens_to_generate, conn):
    generated_token_ids = []
    cache_b = None
    position_ids = None
    first_pass = True
    token_count = 0 
    eos_detected = False
    pass_counter = {"i": 0}
    boundary = stopping_layer - 1

    for i in range(len(model.model.layers)):
        if DEBUG:
            global_idx = i + stopping_layer
            pre_timer, hidden_hook = make_layer_hook(boundary, pass_counter, global_idx)
            layer = model.model.layers[i]
            layer_hooks[i] = (
                layer.register_forward_pre_hook(pre_timer, with_kwargs=True),
                layer.register_forward_hook(hidden_hook),
            )
    
    while eos_detected == False:
        if first_pass:

            #for idx, tensor in layer_outputs_b.items():
                #print(f"Layer {idx} shape after first pass removal: {tensor.shape}")
            
            print("Machine B first pass")
            hidden, position_embeddings, position_ids = receive_handoff(conn, expect=MSG_FIRST_PASS)
            layer_history[pass_counter["i"], stopping_layer] = {
                "position_embeddings": (position_embeddings[0].detach().clone(), position_embeddings[1].detach().clone()),
                "position_ids": position_ids.detach().clone() 
            }

            first_pass = False
            #load file into memory

        else:
            hidden, position_embeddings = receive_handoff(conn, expect=MSG_NEXT_PASS)
            layer_history[pass_counter["i"], stopping_layer] = {
                "position_embeddings": (position_embeddings[0].detach().clone(), position_embeddings[1].detach().clone()),
            }


        print(f"Starting Split 2: Pass #{token_count + 1}")
        token, cache_b = split_2(hidden, position_embeddings, position_ids, model, cache_b)
        #print(hidden.dtype, hidden.device)
        #perform split 2 and generate the next token

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        if token.item() in eos_ids:
            # if we have detect eos/reached token count then we call machine A to start decoding the response by sending eos_detected = True
            eos_detected = True
            send_eos(conn)
            print("Sent EOS Token")
            break

        else:
            generated_token_ids.append(token.item())
            send_token(conn, token)
            token_count += 1
            pass_counter["i"] += 1
            print(f"Sent Token {token_count} \n")
            if token_count >= tokens_to_generate:
                break

    #print(f"layer_outputs_b keys before send: {sorted(layer_outputs_b.keys())}")
    #print(f"layer_outputs_b length: {len(layer_outputs_b)}")
    all_layer_history = {}
    if DEBUG:
        print("Receiving Machine A layer outputs...")
        machine_a_layer_history = receive_layers(conn)

        print("Sending Machine B layer outputs to Machine A...")
        send_layers(conn, layer_history)

        all_layer_history = {**machine_a_layer_history, **layer_history}
        print(len(all_layer_history))
        
        for handles in layer_hooks.values():
            for h in handles:
                h.remove()
    
    ttft = receive_ttft(conn)
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return response, all_layer_history, ttft

# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    conn = setup_machine_b_conn()
    model, tokenizer = setup_model_b(STOPPING_LAYER, MODEL_PATH)
    try:
        response, all_layer_history, ttft = run_machine_b(tokenizer, model, STOPPING_LAYER, TOKENS_TO_GENERATE, conn)
        print("response:", response)
    finally:
        conn.close()