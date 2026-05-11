import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import psutil
import threading
import os
from config import (
    MODEL_PATH,
    PROMPT,
    STOPPING_LAYER,
    TOKENS_TO_GENERATE,
    DEVICE
)

def capture_layers(model, inputs, label, stopping_layer):
    """
    Attaches hooks to every layer and captures:
    - Hidden state output at each layer
    - Time to first token (fires on layer 0's first forward pass)
    - Position embeddings and position ids at the split boundary
    - All timing per layer

    Returns:
        layer_outputs   : dict {layer_idx: hidden_state_tensor}
        handoff_package : dict with hidden, position_ids, position_embeddings
        ttft            : float, time to first token in seconds
        layer_times     : dict {layer_idx: time_taken}
    """
    print(f"Capturing {label} layer outputs...")

    layer_outputs   = {}
    handoff_package = {}
    layer_times     = {}
    ttft_result     = {"ttft": None, "start": time.time(), "fired": False}
    split_boundary  = stopping_layer

    def make_hook(idx):
        def hook_fn(module, input, output):
            t = time.time()

            # Time to first token — fires once on layer 0's first call
            if idx == 0 and not ttft_result["fired"]:
                ttft_result["ttft"]  = t - ttft_result["start"]
                ttft_result["fired"] = True

            hidden = output[0].detach().clone()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            layer_outputs[idx] = hidden
            layer_times[idx]   = time.time() - t

        return hook_fn

    def make_pre_hook(idx):
        def hook_pos(module, args, kwargs):
            # Capture position info at every layer
            # At the split boundary this becomes the handoff package
            pos_emb = kwargs.get("position_embeddings")
            pos_ids = kwargs.get("position_ids")

            if pos_emb is not None and pos_ids is not None:
                cos, sin = pos_emb
                handoff_package["position_embeddings"] = (
                    cos.detach().clone(),
                    sin.detach().clone()
                )
                handoff_package["position_ids"] = pos_ids.detach().clone()

        return hook_pos

    # Register forward hooks and pre hooks on every layer
    hooks = []
    for i in range(len(model.model.layers)):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))
        hooks.append(model.model.layers[i].register_forward_pre_hook(
            make_pre_hook(i), with_kwargs=True
        ))

    # Run one forward pass to populate everything
    with torch.no_grad():
        model(**inputs)

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Build handoff package from the split boundary layer output
    if layer_outputs:
        handoff_package["hidden"] = layer_outputs[split_boundary]

    ttft = ttft_result["ttft"]

    # Print layer timing summary
    print(f"\n{'='*55}")
    print(f"{label} — Layer Capture Summary")
    print(f"{'='*55}")
    print(f"{'Layer':<8} {'Hidden Shape':<22} {'Time (ms)':<12}")
    print(f"{'-'*55}")
    for idx in sorted(layer_outputs.keys()):
        shape   = str(tuple(layer_outputs[idx].shape))
        elapsed = layer_times.get(idx, 0) * 1000
        print(f"{idx:<8} {shape:<22} {elapsed:<12.3f}")
    print(f"{'-'*55}")
    print(f"Time to first token: {ttft:.3f}s")
    print(f"Layers captured:     {len(layer_outputs)}")
    print(f"Handoff package keys: {list(handoff_package.keys())}")
    print(f"{'='*55}\n")

    return layer_outputs, handoff_package, ttft, layer_times


def default_generation(model_path, prompt, stopping_layer, tokens_to_generate):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=DEVICE
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

    # Capture everything in one call
    layer_outputs, handoff_package, ttft, layer_times = capture_layers(
        model, inputs, "Full Generation", stopping_layer
    )

    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            use_cache=True
        )
    gen_time = time.time() - gen_start

    output_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Generation time:     {gen_time:.2f}s")
    print(f"Time to first token: {ttft:.3f}s")
    print(f"Response: {output_response}")

    return {
        "response":       output_response,
        "layer_outputs":  layer_outputs,
        "handoff_package": handoff_package,
        "ttft":           ttft,
        "layer_times":    layer_times,
        "gen_time":       gen_time,
    }

if __name__ == "__main__":
    default_generation(MODEL_PATH, PROMPT, STOPPING_LAYER, TOKENS_TO_GENERATE)
    