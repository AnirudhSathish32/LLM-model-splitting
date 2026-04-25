from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
#Loading model 
tokenizer = AutoTokenizer.from_pretrained(model_path)
#Loading Tokenizer
messages = [
    {"role": "user", "content": "how many planets are in the solar system"}
]
# Loading context
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

#Loading prompt
inputs = tokenizer(prompt, return_tensors="pt")
#Tokenizing prompt into input tensors
model.eval()
#neural network enters evaluation mode so it behaves predictably

stopping_layer = 14
starting_layer = stopping_layer + 1
tokens_to_generate = 200


def perform_full_generation():
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return print(response)

def perform_split_generation(tokens_to_generate):
    generated_token_ids = []
    
    # Start with the original input ids
    current_input_ids = inputs["input_ids"]
    
    for _ in range(tokens_to_generate):
        
        # ---- Machine A: layers 1-14 ----
        captured = {}
        def hook_fn(module, input, output):
            hidden = output[0].detach().clone()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            captured["hidden"] = hidden
            raise StopIteration
        
        def hook_pos(module, args, kwargs):
            cos, sin = kwargs.get("position_embeddings")
            captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
            captured["position_ids"] = kwargs.get("position_ids")

        h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
        h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)

        try:
            with torch.no_grad():
                model(input_ids=current_input_ids)
        except StopIteration:
            pass

        h1.remove()
        h2.remove()

        # ---- Machine B: layers 15-28 ----
        with torch.no_grad():
            x = captured["hidden"]
            for i in range(starting_layer - 1, len(model.model.layers)):
                x = model.model.layers[i](
                    x,
                    position_ids=captured["position_ids"],
                    position_embeddings=captured["position_embeddings"],
                )[0]
                if x.dim() == 2:
                    x = x.unsqueeze(0)

            x = model.model.norm(x)
            logits = model.lm_head(x)

        # ---- Pick next token ----
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated_token_ids.append(next_token_id.item())

        # ---- Check if model is done ----
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if next_token_id.item() in eos_ids:
            break

        # ---- Append new token to input for next pass ----
        current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)

    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return print(response)

if __name__ == "__main__":
    print("=====================FULL GENERATION============================")
    perform_full_generation()
    print("=====================SPLIT GENERATION============================")
    perform_split_generation(tokens_to_generate)