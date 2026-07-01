from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DynamicCache
from accelerate import init_empty_weights
from safetensors.torch import load_file
import torch.nn as nn
import os
import time
import torch
import gc

class Model:
    def __init__(self, model_name, role, layer_start, layer_end, local_config, dtype):
        self.model_name  = model_name
        self.role        = role
        self.is_master   = role == "master"
        self.is_tail     = role == "tail"
        self.layer_start = layer_start
        self.layer_end   = layer_end
        self.device      = local_config.device
        self.dtype       = dtype
        self.model_path   = local_config.model_path
        self.layers_path  = local_config.layers_path
        self.cache = None

        self.model     = None
        self.tokenizer = None
        self.config    = None

        # per-turn state
        self.handoff_package = {}
        self.layer_history   = {}
        self.timing_starts   = {}
        self.pass_counter    = {"i": 0}
        self.layer_hooks     = {}
        self.generated_ids   = []

    def reset_turn_state(self):
        self.handoff_package = {}
        self.layer_history = {}
        self.timing_starts = {}
        self.pass_counter = {"i": 0}
        self.generated_ids = []

    def split_tail(self, hidden, model, cache_tail=None):
        """
        ---- Machine B ----
        Second Split 
        """
    
        if cache_tail is None:
            cache_tail = DynamicCache()
            #for _ in range(len(model.model.layers)):
            #   cache_b.layers.append(DynamicLayer())
            print(f"  [split_2] fresh cache, type={type(cache_tail)}, has get_seq_length={hasattr(cache_tail, 'get_seq_length')}")

        with torch.no_grad():
            x =  model.model(
                inputs_embeds=hidden,
                past_key_values=cache_tail,
                use_cache=True,
            )
            print(f"  [split_tail] layer0 returned type={type(x)}, len={len(x) if isinstance(x, tuple) else 'n/a'}")
            print(f"  [split_tail] cache len AFTER layer0 = {cache_tail.get_seq_length()}")
            print(f"  [split_tail] layer0 keys is None? {cache_tail.layers[0].keys is None if cache_tail.layers else 'no layers'}")

            x = x.last_hidden_state
            logits = model.lm_head(x)

            # ---- Pick next token ----
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

        return  next_token_id, cache_tail

    def split_master(self, current_input_ids, model, cache_master=None):
        """
        ---- Machine A ----
        First Split

        """
        try:
            with torch.no_grad():
                model(input_ids=current_input_ids,
                    past_key_values=cache_master,
                    use_cache=True,
                    return_dict=True
                    )
        except StopIteration:
            pass
        hidden = self.handoff_package["hidden"]
        position_embeddings = self.handoff_package["position_embeddings"]
        position_ids = self.handoff_package["position_ids"]
        cache_master = self.handoff_package["cache_master"]

        return hidden, position_embeddings, position_ids, cache_master
    
    def make_layer_hook(self, boundary, pass_counter, global_idx):
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
            self.timing_starts[key] = time.perf_counter()


        def hidden_hook(module, input, output):
            key = (pass_counter["i"], global_idx)
            t0 = self.timing_starts.get(key)
            
            if t0 is not None:
                dur = time.perf_counter() - t0
            else:
                dur = 0.0 
            
            hidden = output[0].detach()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            # Validation capture — every layer
            self.layer_history[key] = {
                "hidden": hidden,
                "dur": dur,
            }

            # Boundary layer — this is the handoff point
            if is_boundary:
                self.handoff_package["hidden"] = hidden
                raise StopIteration   # halt forward pass — Machine A is done

        return timer_start, hidden_hook
    
    """
    def positional_hook(module, args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        self.handoff_package["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        self.handoff_package["position_ids"] = kwargs.get("position_ids")
        self.handoff_package["cache_a"] = kwargs.get("past_key_values")
    """

    def setup_model_master(self, stopping_layer:int, model_path, prompt):
        start = time.time()
        model_path = model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        config.num_hidden_layers = stopping_layer

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        
        model_name = os.path.basename(model_path)
        layers_dir = f"./layers/{model_name}"
        state_a = {}

        state_a.update(load_file(f"{layers_dir}/embed_tokens.safetensors", device=self.device))
        for i in range(stopping_layer):
            state_a.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=self.device))
            print(f"Loaded layer {i}")

        model.load_state_dict(
            state_a,
            strict=False,
            assign=True
        )

        model.eval()

        # Prompt setup lives on Machine A — it drives the generation loop
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        print(f"Load time: {time.time() - start:.2f}s \n")
        print("Machine A ready \n")

        return model, inputs, tokenizer

    def setup_model_tail(self, stopping_layer:int, model_path):
        start = time.time()

        model_path = model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        original_total_layers = config.num_hidden_layers 

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model_name = os.path.basename(model_path)
        layers_dir = f"./layers/{model_name}"
        state_b = {}

        for i in range(stopping_layer, original_total_layers):
            state_b.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=self.device))
            print(f"Loaded layer {i}")

        state_b.update(load_file(f"{layers_dir}/norm.safetensors", device=self.device))
        state_b.update(load_file(f"{layers_dir}/head.safetensors", device=self.device))

        model.load_state_dict(
            state_b,
            strict=False,
            assign=True
        )
        

        kept_layers = model.model.layers[stopping_layer:]
        model.model.layers = nn.ModuleList(kept_layers)
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
        
        #for i, layer in enumerate(model.model.layers):
            #print(i, layer.input_layernorm.weight.device)

        model.eval()

        print(f"Load time: {time.time() - start:.2f}s \n")
        print("Machine B ready \n")

        return model, tokenizer

    def setup_model_middle(self, layer_start, layer_end, model_path, device):
        start = time.time()

        config = AutoConfig.from_pretrained(model_path)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model_name = os.path.basename(model_path)
        layers_dir = f"./layers/{model_name}"
        state = {}

        for i in range(layer_start, layer_end + 1):
            state.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=device))
            print(f"Loaded layer {i}")

        model.load_state_dict(state, strict=False, assign=True)

        kept_layers = model.model.layers[layer_start:layer_end + 1]
        model.model.layers = nn.ModuleList(kept_layers)
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i

        model.eval()
        print(f"Middle node ready — layers {layer_start}..{layer_end}, "
            f"load time: {time.time() - start:.2f}s")

        return model
    
    def unload_model(self):
        self.remove_hooks()
        
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.config = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.reset_turn_state() 

    def remove_hooks(self):
        """Remove all registered hooks from the model's layers."""
        for handles in self.layer_hooks.values():
            if isinstance(handles, tuple):
                for h in handles:
                    h.remove()
            else:
                handles.remove()
        self.layer_hooks = {}
    