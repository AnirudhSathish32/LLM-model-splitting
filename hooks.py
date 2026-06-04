

captured = {}

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