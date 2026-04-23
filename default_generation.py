from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

#model.eval()

prompt = "Hello World"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=False))