import machine_a
import default_generation

import torch
import time
import os
import psutil
import math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache, DynamicLayer
from safetensors.torch import load_file

# =======================================================================
# CONFIG / INITIALIZATION
# ======================================================================
model_path = "./llama-3b"
stopping_layer = 14
starting_layer = stopping_layer + 1
prompt = "Hello world"
tokens_to_generate = 50 

setup 