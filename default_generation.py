from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import psutil
import time
import threading
import os

# ================MEMORY MONITORING===============

process = psutil.Process(os.getpid())
peak_ram = 0
monitoring = True

def get_ram():
    return process.memory_info().rss / (1024 ** 3)

def monitor_memory():
    global peak_ram
    while monitoring:
        current = get_ram()
        if current > peak_ram:
            peak_ram = current
        time.sleep(0.25)

#======================================================
start = time.time()

# =========LOADING==============
load_start = time.time()


model_path = "./llama-8b"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32, 
    output_hidden_states=True)

load_end = time.time()

print(f"Model load time: {load_end - load_start:.2f} sec")
print(f"RAM after model load: {get_ram():.2f} GB")

# ======================TOKENIZING=================
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompt = "Hello World"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"RAM before generation: {get_ram():.2f} GB")

# START MEMORY MONITOR

monitor_thread = threading.Thread(target=monitor_memory)
monitor_thread.start()

# START GENERATION

gen_start = time.time()


with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7
    )
    
gen_end = time.time()

# STOP MONITORING

monitoring = False
monitor_thread.join()
end = time.time()


output_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
generation_time = gen_end - gen_start
tokens_per_sec = generated_tokens / generation_time if generation_time > 0 else 0
end = time.time()

print("=================OUTPUT====================")
print(output_response)


print("=================METRICS====================")
print(f"Generated tokens:      {generated_tokens}")
print(f"Generation time:       {generation_time:.2f} sec")
print(f"Tokens / second:       {tokens_per_sec:.4f}")
print(f"Peak RAM usage:        {peak_ram:.2f} GB")
print(f"Final RAM usage:       {get_ram():.2f} GB")
print(f"Total time: {end - start:.2f} seconds")