import inspect
from transformers.models.llama import modeling_llama

# the attention module is what calls cache.update() — find its real signature + body
print("=== Attention class ===")
attn_cls = modeling_llama.LlamaAttention
print(inspect.signature(attn_cls.forward))
print("\n=== Attention forward source ===")
print(inspect.getsource(attn_cls.forward))