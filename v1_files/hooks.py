import time 

handoff_package = {}
layer_history = {}
layer_hooks = {}
timing_starts = {}


def make_layer_hook(boundary, pass_counter, global_idx):
    """
    Single unified forward hook for each layer.

    Behavior depends on position:
    - Every layer: capture hidden state for validation 
    - Boundary layer (stopping_layer - 1): also save handoff hidden + raise StopIteration

    idx               : this layer's index
    stopping_layer    : the split boundary
    capture_validation: whether to record validation data (first pass only)
    """
    
    is_boundary = global_idx == boundary
    

    def timer_start(module, args, kwargs):
        key = (pass_counter["i"], global_idx)
        timing_starts[key] = time.perf_counter()


    def hidden_hook(module, input, output):
        key = (pass_counter["i"], global_idx)
        t0 = timing_starts.get(key)
        
        if t0 is not None:
            dur = time.perf_counter() - t0
        else:
            dur = 0.0 
        
        hidden = output[0].detach()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Validation capture — every layer
        layer_history[key] = {
            "hidden": hidden,
            "dur": dur,
        }

        # Boundary layer — this is the handoff point
        if is_boundary:
            handoff_package["hidden"] = hidden
            raise StopIteration   # halt forward pass — Machine A is done

    return timer_start, hidden_hook, 
    
def positional_hook(module, args, kwargs):
    cos, sin = kwargs.get("position_embeddings")
    handoff_package["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
    handoff_package["position_ids"] = kwargs.get("position_ids")
    handoff_package["cache_a"] = kwargs.get("past_key_values")

