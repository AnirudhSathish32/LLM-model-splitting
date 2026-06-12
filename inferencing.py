from transformers import DynamicCache, DynamicLayer
import torch 
import os
from config import (
    DEVICE,
    RECEIVED_DIR,
    HANDOFF_DIR
)

from hooks import (
    handoff_package
)

def load_handoff_package(save_dir=RECEIVED_DIR, first_pass=True):
    if first_pass:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=DEVICE)
        cos = torch.load(f"{save_dir}/cos.pt", map_location=DEVICE)
        sin = torch.load(f"{save_dir}/sin.pt", map_location=DEVICE)
        position_embeddings = (cos, sin)
        position_ids = torch.load(f"{save_dir}/position_ids.pt", map_location=DEVICE)
        return hidden, position_embeddings, position_ids
    else:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=DEVICE)
        return hidden
    
def save_handoff_package(hidden, position_embeddings, position_ids, save_dir=HANDOFF_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden, f"{save_dir}/hidden.pt")
    torch.save(position_embeddings[0], f"{save_dir}/cos.pt")
    torch.save(position_embeddings[1], f"{save_dir}/sin.pt")
    torch.save(position_ids, f"{save_dir}/position_ids.pt")


def split_2(hidden, position_embeddings, position_ids, model, cache_b=None):
    """
    ---- Machine B ----
    Second Split 
    """
    
    if cache_b is None:
        cache_b = DynamicCache()
        #for _ in range(len(model.model.layers)):
         #   cache_b.layers.append(DynamicLayer())
        print(f"  [split_2] fresh cache, type={type(cache_b)}, has get_seq_length={hasattr(cache_b, 'get_seq_length')}")

    with torch.no_grad():
        x =  model.model(
            inputs_embeds=hidden,
            past_key_values=cache_b,
            use_cache=True,
        )
        print(f"  [split_2] layer0 returned type={type(x)}, len={len(x) if isinstance(x, tuple) else 'n/a'}")
        print(f"  [split_2] cache len AFTER layer0 = {cache_b.get_seq_length()}")
        print(f"  [split_2] layer0 keys is None? {cache_b.layers[0].keys is None if cache_b.layers else 'no layers'}")

        x = x.last_hidden_state
        logits = model.lm_head(x)

        # ---- Pick next token ----
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

    return  next_token_id, cache_b

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
                return_dict=True
                )
    except StopIteration:
        pass
    hidden = handoff_package["hidden"]
    position_embeddings = handoff_package["position_embeddings"]
    position_ids = handoff_package["position_ids"]
    cache_a = handoff_package["cache_a"]

    return hidden, position_embeddings, position_ids, cache_a