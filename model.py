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
        self.model_path  = local_config.model_path
        self.layers_path = local_config.layers_path

        self.model     = None
        self.tokenizer = None
        self.config    = None

        # per-turn state (reset between generation calls)
        self.handoff_package = {}
        self.layer_history   = {}
        self.timing_starts   = {}
        self.pass_counter    = {"i": 0}
        self.layer_hooks     = {}
        self.generated_ids   = []

    # ================================================================
    # LIFECYCLE
    # ================================================================

    def load(self):
        """
        Single entry point for loading the model slice.

        Master:  embed_tokens + layers 0..layer_end.
                 Config is patched to num_hidden_layers = layer_end + 1
                 so the model is created with exactly the right count.
                 No slicing needed. Tokenizer loaded.

        Worker:  layers layer_start..layer_end only.
                 Full config skeleton, then slice + re-index.
                 model.model.norm replaced with Identity (only tail norms).

        Tail:    layers layer_start..layer_end + norm + lm_head.
                 Slice + re-index. Tokenizer loaded.
        """
        start = time.time()
        model_dir = os.path.join(self.model_path, self.model_name)
        layers_dir = os.path.join(self.layers_path, self.model_name)

        self.config = AutoConfig.from_pretrained(model_dir)

        if self.is_master:
            self._load_master(model_dir, layers_dir)
        elif self.is_tail:
            self._load_tail(model_dir, layers_dir)
        else:
            self._load_worker(model_dir, layers_dir)

        n_layers = self.layer_end - self.layer_start + 1
        elapsed = time.time() - start
        print(f"[Model] {self.role} loaded: layers {self.layer_start}..{self.layer_end} "
              f"({n_layers} layers) in {elapsed:.2f}s")

    def _load_master(self, model_dir, layers_dir):
        """Master owns embed + first N layers. Uses model() for forward."""
        load_config = AutoConfig.from_pretrained(model_dir)
        load_config.num_hidden_layers = self.layer_end + 1

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(load_config)

        state = {}
        state.update(load_file(f"{layers_dir}/embed_tokens.safetensors", device=self.device))
        for i in range(self.layer_start, self.layer_end + 1):
            state.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=self.device))
            print(f"  Loaded layer {i}")

        self.model.load_state_dict(state, strict=False, assign=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def _load_tail(self, model_dir, layers_dir):
        """Tail owns last N layers + norm + lm_head. Uses model.model() + lm_head."""
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)

        state = {}
        for i in range(self.layer_start, self.layer_end + 1):
            state.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=self.device))
            print(f"  Loaded layer {i}")
        state.update(load_file(f"{layers_dir}/norm.safetensors", device=self.device))
        state.update(load_file(f"{layers_dir}/head.safetensors", device=self.device))

        self.model.load_state_dict(state, strict=False, assign=True)

        # Slice to keep only our layers, re-index for cache
        kept = self.model.model.layers[self.layer_start:self.layer_end + 1]
        self.model.model.layers = nn.ModuleList(kept)
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.layer_idx = i

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def _load_worker(self, model_dir, layers_dir):
        """Worker owns middle layers only. Uses model.model() with Identity norm."""
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)

        state = {}
        for i in range(self.layer_start, self.layer_end + 1):
            state.update(load_file(f"{layers_dir}/layer_{i}.safetensors", device=self.device))
            print(f"  Loaded layer {i}")

        self.model.load_state_dict(state, strict=False, assign=True)

        kept = self.model.model.layers[self.layer_start:self.layer_end + 1]
        self.model.model.layers = nn.ModuleList(kept)
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.layer_idx = i

        # Worker must NOT apply norm — only tail does that
        self.model.model.norm = nn.Identity()

        self.model.eval()

    def unload(self):
        """Free all model resources."""
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

    def reset_turn_state(self):
        """Clear per-generation-turn state. Called between queries."""
        self.handoff_package = {}
        self.layer_history = {}
        self.timing_starts = {}
        self.pass_counter = {"i": 0}
        self.generated_ids = []

    

    # ================================================================
    # FORWARD DISPATCH
    # ================================================================

    def forward(self, model_input, cache):
        """
        Unified forward call. Dispatches based on role.

        Args:
            model_input:  input_ids tensor (master) or hidden state tensor (worker/tail)
            cache:        DynamicCache or None (created fresh if None)

        Returns:
            master:  (hidden_state, cache)
            worker:  (hidden_state, cache)
            tail:    (next_token_id, cache)
        """
        if self.is_master:
            return self._forward_master(model_input, cache)
        elif self.is_tail:
            return self._forward_tail(model_input, cache)
        else:
            return self._forward_worker(model_input, cache)

    def _forward_master(self, input_ids, cache):
        """
        Run model(input_ids=...) with hooks. The boundary layer hook
        captures the hidden state and raises StopIteration to halt
        the forward pass early. Cache is mutated in-place.
        """
        if cache is None:
            cache = DynamicCache()

        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
        except StopIteration:
            pass

        hidden = self.handoff_package["hidden"]
        return hidden, cache

    def _forward_worker(self, hidden, cache):
        """
        Run model.model(inputs_embeds=...) — inner model without embed/head.
        Norm is Identity so hidden passes through unnormed.
        """
        if cache is None:
            cache = DynamicCache()

        with torch.no_grad():
            out = self.model.model(
                inputs_embeds=hidden,
                past_key_values=cache,
                use_cache=True,
                attn_implementation="eager",
            )

        return out.last_hidden_state, cache

    def _forward_tail(self, hidden, cache):
        """
        Run model.model(inputs_embeds=...) + lm_head to produce next token.
        Norm IS applied here (not replaced with Identity).
        """
        if cache is None:
            cache = DynamicCache()

        with torch.no_grad():
            out = self.model.model(
                inputs_embeds=hidden,
                past_key_values=cache,
                use_cache=True,
                attn_implementation="eager",
            )
            logits = self.model.lm_head(out.last_hidden_state)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

        return next_token_id, cache

    # ================================================================
    # TOKENIZATION (master and tail only)
    # ================================================================

    def tokenize(self, messages):
        """
        Apply chat template to message list, return input_ids tensor.
        Used by master to prepare the prompt.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.input_ids

    def decode(self, token_ids):
        """Decode a list of token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ================================================================
    # HOOKS (master only — StopIteration at boundary layer)
    # ================================================================

    def register_hooks(self, debug=False):
        """
        Register forward hooks on master's layers.
        The boundary layer (layer_end) gets a hook that captures hidden
        state and raises StopIteration to halt the forward pass.

        Workers and tail don't need hooks — they use model.model() directly.
        """
        if not self.is_master:
            return

        for i, layer in enumerate(self.model.model.layers):
            global_idx = self.layer_start + i
            timer_start, hidden_hook = self._make_layer_hook(
                boundary=self.layer_end,
                global_idx=global_idx,
            )
            pre = layer.register_forward_pre_hook(timer_start, with_kwargs=True)
            post = layer.register_forward_hook(hidden_hook)
            self.layer_hooks[global_idx] = (pre, post)

        print(f"[Model] Registered hooks on layers {self.layer_start}..{self.layer_end}, "
              f"boundary={self.layer_end}")

    def _make_layer_hook(self, boundary, global_idx):
        """
        Create pre/post hooks for a single layer.

        Every layer: captures hidden + timing for validation.
        Boundary layer: also raises StopIteration to halt forward pass.
        """
        is_boundary = (global_idx == boundary)
        pass_counter = self.pass_counter

        def timer_start(module, args, kwargs):
            key = (pass_counter["i"], global_idx)
            self.timing_starts[key] = time.perf_counter()

        def hidden_hook(module, input, output):
            key = (pass_counter["i"], global_idx)
            t0 = self.timing_starts.get(key)
            dur = (time.perf_counter() - t0) if t0 is not None else 0.0

            # Transformers 5.x: LlamaDecoderLayer returns bare tensor, not tuple
            if isinstance(output, tuple):
                hidden = output[0].detach()
            else:
                hidden = output.detach()

            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            self.layer_history[key] = {"hidden": hidden, "dur": dur}

            if is_boundary:
                self.handoff_package["hidden"] = hidden
                raise StopIteration

        return timer_start, hidden_hook

    def remove_hooks(self):
        """Remove all registered hooks from the model's layers."""
        for handles in self.layer_hooks.values():
            if isinstance(handles, tuple):
                for h in handles:
                    h.remove()
            else:
                handles.remove()
        self.layer_hooks = {}
