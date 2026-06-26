import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import psutil
import threading
import os
from config import (
    MODEL_PATH,
    DTYPE,
    PROMPT,
    STOPPING_LAYER,
    TOKENS_TO_GENERATE,
    DEVICE
)

def add_validation_hooks(model, layer_history, timing_starts, pass_counter, ttft_holder):
    """
    Attach persistent per-layer hooks to the FULL (single-machine) model.
 
    Captures hidden state + per-layer duration for EVERY layer on EVERY token pass,
    keyed (pass, layer_idx) so the result lines up 1:1 with the split machines'
    layer_history (Machine A keys 0..boundary, Machine B keys boundary+1..N-1).
 
    Pass tracking: generate() owns its own decode loop, so we can't bump the pass
    from outside. Instead the layer-0 PRE hook bumps pass_counter each time a fresh
    forward begins. pass_counter starts at -1, so the prefill forward becomes pass 0 —
    matching the split side, where pass 0 is also the prefill.
 
    This hook does NOT raise StopIteration; the full model must run to completion.
 
    Returns the list of handles for teardown.
    """
    handles = []
    n    = len(model.model.layers)
    last = n - 1
 
    def make_pre(idx):
        def pre(module, args, kwargs):
            if idx == 0:
                pass_counter["i"] += 1          # a new token step is starting
            timing_starts[(pass_counter["i"], idx)] = time.perf_counter()
        return pre
 
    def make_post(idx):
        def post(module, input, output):
            key = (pass_counter["i"], idx)
            t0  = timing_starts.get(key)
            dur = (time.perf_counter() - t0) if t0 is not None else 0.0
 
            hidden = output[0].detach().clone()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
 
            layer_history[key] = {"hidden": hidden, "dur": dur}
 
            # TTFT ≈ end of the prefill forward (pass 0, last layer), when the first
            # token's logits become available.
            if idx == last and ttft_holder["ttft"] is None:
                ttft_holder["ttft"] = time.perf_counter() - ttft_holder["start"]
        return post
 
    for i in range(n):
        layer = model.model.layers[i]
        handles.append(layer.register_forward_pre_hook(make_pre(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post(i)))
    return handles


def default_generation(model_path, prompt, stopping_layer, tokens_to_generate):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=DEVICE,
        dtype=DTYPE,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model.eval()

    layer_history = {}
    timing_starts = {}
    pass_counter  = {"i": -1}                         # first layer-0 pre-hook -> pass 0
    ttft_holder   = {"ttft": None, "start": None}
 
    handles = add_validation_hooks(model, layer_history, timing_starts, pass_counter, ttft_holder)
 
    ttft_holder["start"] = time.perf_counter()
    gen_start = time.perf_counter()

    gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            use_cache=True,
            do_sample=False
        )
    gen_time = time.time() - gen_start

    for h in handles:
        h.remove()

    input_len = inputs["input_ids"].shape[1]

    output_response = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    )
    ttft = ttft_holder["ttft"]

    print(f"Generation time:     {gen_time:.2f}s")
    print(f"Time to first token: {ttft:.3f}s")
    print(f"Response: {output_response}")
    print(f"layer history: {layer_history}")

    return {
        "model":          model,
        "response":       output_response,
        "layer_history":  layer_history,
        "ttft":           ttft,
        "gen_time":       gen_time,
    }

if __name__ == "__main__":
    default_generation(MODEL_PATH, PROMPT, STOPPING_LAYER, TOKENS_TO_GENERATE)
    