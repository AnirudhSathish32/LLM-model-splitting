import torch
import os

# ================================================================
# EVERYTHING BUT DEVICE, HANDOFF_DIR, LAYERS_DIR AND RECEIVED_DIR MUST BE SAME
# ACROSS MACHINE_A AND MACHINE_B
# ================================================================

# ================================================================
# MODEL CONFIG
# ================================================================
MODEL_PATH     = "./llama-3b"
STOPPING_LAYER = 16
DTYPE = torch.float16

# ================================================================
# GENERATION CONFIG
# ================================================================
PROMPT             = ("Act as a legal strategist." "List 5 high-stakes considerations for drafting commercial "
                    "contracts to minimize risk and ensure performance. Use one sentence per point.")
TOKENS_TO_GENERATE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# NETWORK CONFIG 
# ================================================================
MACHINE_A_TAILSCALE_IP = "100.74.100.92" 
TAILSCALE_PORT         = 65432
MSG_FIRST_PASS = 1
MSG_NEXT_PASS  = 2
MSG_TOKEN      = 3
MSG_EOS        = 4
MSG_LAYER      = 5
MSG_TTFT       = 6
MSG_STOP       = 7

ANIRUDH_MACHINE_A = "100.74.100.92"
PRANATHI_MACHINE_A = ""

# ================================================================
# PATHS
# ================================================================
HANDOFF_DIR  = "./handoff"
RECEIVED_DIR = "./received"
LAYERS_DIR   = f"./layers/{os.path.basename(MODEL_PATH)}"