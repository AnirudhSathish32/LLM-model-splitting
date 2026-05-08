from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
inputs = tokenizer("Hello world", return_tensors="pt")

captured = {}

def hook_hidden(module, input, output):
    hidden = output[0].detach().clone()
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    captured["hidden_state"] = hidden

def hook_pos(module, args, kwargs):
    cos, sin = kwargs.get("position_embeddings")
    captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
    captured["position_ids"] = kwargs.get("position_ids")

handle1 = model.model.layers[13].register_forward_hook(hook_hidden)
handle2 = model.model.layers[14].register_forward_pre_hook(hook_pos, with_kwargs=True)

with torch.no_grad():
    full_output = model(**inputs)

handle1.remove()
handle2.remove()

print("cos shape:", captured["position_embeddings"][0].shape)
print("sin shape:", captured["position_embeddings"][1].shape)
print("hidden state shape:", captured["hidden_state"].shape)

with torch.no_grad():
    x = captured["hidden_state"]
    pos_ids = captured["position_ids"]
    pos_emb = captured["position_embeddings"]

    for i in range(14, len(model.model.layers)):
        result = model.model.layers[i](
            x,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
        )
        x = result[0]
        # Restore batch dimension if layer squeezed it
        if x.dim() == 2:
            x = x.unsqueeze(0)

    x = model.model.norm(x)
    logits_partial = model.lm_head(x)

next_token_full = torch.argmax(full_output.logits[:, -1, :], dim=-1)
next_token_partial = torch.argmax(logits_partial[:, -1, :], dim=-1)

print("Full pass next token:    ", tokenizer.decode(next_token_full))
print("Partial pass next token: ", tokenizer.decode(next_token_partial))
print("Tokens match:", next_token_full.item() == next_token_partial.item())
print("Logit diff (max):", (full_output.logits - logits_partial).abs().max().item())