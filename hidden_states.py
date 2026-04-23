from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

prompt = "Hello World"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

layer = 4

hidden_states = outputs.hidden_states # per layer interpretation 
logits = outputs.logits # final output logits
attentions = outputs.attentions # per layer attention weights produced

print(hidden_states[layer])
#print(logits)
#print(attentions)
#print("Number of layers:", len(hidden_states))
